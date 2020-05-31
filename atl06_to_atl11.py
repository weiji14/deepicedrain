# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: deepicedrain
#     language: python
#     name: deepicedrain
# ---

# %% [markdown]
# # **ATL06 to ATL11**
#
# Converting the ICESat-2 ATL06 (Land Ice Height) product to ATL11 (Land Ice Height Changes).
# Also convert the ATL11 file format from HDF5 to [Zarr](https://zarr.readthedocs.io/).

# %%
import os
import glob
import sys
import subprocess

import dask
import dask.distributed
import h5py
import intake
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import tqdm
import xarray as xr
import zarr

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# %%
client = dask.distributed.Client(n_workers=64, threads_per_worker=1)
client

# %%
# Create ATL06_to_ATL11 processing script, if not already present
if not os.path.exists("ATL06_to_ATL11_Antarctica.sh"):
    # find number of cycles for each reference ground track and each orbital segment
    func = lambda ref_gt, orb_st: len(
        glob.glob(f"ATL06.003/**/ATL06*_*_{ref_gt:04d}??{orb_st}_*.h5")
    )
    futures = []
    for referencegroundtrack in range(1387, 0, -1):
        for orbitalsegment in [10, 11, 12]:  # loop through Antarctic orbital segments
            numcycles = client.submit(
                func,
                referencegroundtrack,
                orbitalsegment,
                key=f"{referencegroundtrack:04d}-{orbitalsegment}",
            )
            futures.append(numcycles)

    # Prepare string to write into ATL06_to_ATL11_Antarctica.sh bash script
    writelines = []
    for f in tqdm.tqdm(
        iterable=dask.distributed.as_completed(futures=futures), total=len(futures)
    ):
        referencegroundtrack, orbitalsegment = f.key.split("-")
        cycles = f.result()
        writelines.append(
            f"python3 ATL11/ATL06_to_ATL11.py"
            f" {referencegroundtrack} {orbitalsegment}"
            f" --cycles 01 {cycles:02d}"
            f" --Release 3"
            f" --directory 'ATL06.003/**/'"
            f" --out_dir ATL11.001\n",
        )
    writelines.sort()  # sort writelines in place

    # Finally create the bash script
    with open(file="ATL06_to_ATL11_Antarctica.sh", mode="w") as f:
        f.writelines(writelines)


# %% [markdown]
# Now use [GNU parallel](https://www.gnu.org/software/parallel/parallel_tutorial.html) to run the script in parallel.
# Will take about 1 week to run on 64 cores.
#
# Reference:
#
# - O. Tange (2018): GNU Parallel 2018, Mar 2018, ISBN 9781387509881, DOI https://doi.org/10.5281/zenodo.1146014

# %%
# !PYTHONPATH=`pwd` PYTHONWARNINGS="ignore" parallel -a ATL06_to_ATL11_Antarctica.sh --bar --results logdir --joblog log --jobs 64 > /dev/null

# %% [markdown]
# ## Convert from HDF5 to Zarr format
#
# For faster data access speeds!
# We'll collect the data for each Reference Ground Track,
# and store it inside a Zarr format.
#
# Grouping hierarchy:
#   - Reference Ground Track (1-1387)
#     - Orbital Segments (10, 11, 12)
#       - Laser Pairs (pt1, pt2, pt3)
#         - Attributes (longitude, latitude, h_corr, delta_time, etc)

# %%
# for atl11file in tqdm.tqdm(iterable=sorted(glob.glob("ATL11.001/*.h5"))):
#     name = os.path.basename(p=os.path.splitext(p=atl11file)[0])

max_cycles: int = max([int(f[-12:-11]) for f in glob.glob("ATL11.001/*.h5")])
print(f"{max_cycles} ICESat-2 cycles available")

# %%
@dask.delayed
def open_ATL11(atl11file: str, group: str) -> xr.Dataset:
    """
    Opens up an ATL11 file using xarray and does some light pre-processing:
    - Mask values using _FillValue ??
    - Convert attribute format from binary to str
    - Rename 'quality_summary' to f'quality_summary_{subgroup}'
    """
    pair, subgroup = group.split("/")
    ds = xr.open_dataset(
        filename_or_obj=atl11file,
        group=f"{pair}/{subgroup}",
        engine="h5netcdf",
        mask_and_scale=True,
    )

    # Change xarray.Dataset attributes from binary to str type
    # fixes issue when saving to Zarr format later
    # TypeError: Object of type bytes is not JSON serializable
    for key, variable in ds.variables.items():
        assert isinstance(ds[key].DIMENSION_LABELS, np.ndarray)
        ds[key].attrs["DIMENSION_LABELS"] = (
            ds[key].attrs["DIMENSION_LABELS"].astype(str)
        )

    # Rename quality_summary variable to avoid name class when merging
    ds = ds.rename({"quality_summary": f"quality_summary_{subgroup}"})

    return ds


# %%
# Consolidate together Antarctic orbital segments 10, 11, 12 into one file
# Also consolidate all three laser pairs pt1, pt2, pt3 into one file
atl11_dict = {}
for rgt in tqdm.trange(1387):
    atl11files: list = glob.glob(f"ATL11.001/ATL11_{rgt+1:04d}1?_????_??_v00?.h5")

    # Manually handle exceptional cases
    try:
        assert len(atl11files) == 3  # Should be 3 files for Orbital Segments 10,11,12
    except AssertionError:
        if (len(atl11files) == 0 and rgt + 1 in [47, 214]) or (
            len(atl11files) == 2
            and rgt + 1 in [31, 54, 73, 100, 106, 161, 603, 915, 1045, 1106, 1151]
        ):
            pass
        else:
            raise
    # Note ["ATL11.001/ATL11_014512_0103_03_v001.h5", "ATL11.001/ATL11_115810_0104_03_v001.h5"]
    # are missing pt2 and pt3 groups

    if atl11files:
        pattern: dict = intake.source.utils.reverse_format(
            format_string="ATL11.001/ATL11_{referencegroundtrack:4}{orbitalsegment:2}_{cycles:4}_{revision:2}_v{version:3}.h5",
            resolved_string=sorted(atl11files)[1],
        )
        zarrfilepath: str = "ATL11.001z123/ATL11_{referencegroundtrack}1x_{cycles}_{revision}_v{version}.zarr".format(
            **pattern
        )
        atl11_dict[zarrfilepath] = atl11files


# %%
# Gather up all the dask.delayed conversion tasks to store data into Zarr!
stores = []
for zarrfilepath, atl11files in tqdm.tqdm(iterable=atl11_dict.items()):
    zarr.open(store=zarrfilepath, mode="w")  # Make a new file/overwrite existing
    datasets = []
    for atl11file in atl11files:  # Orbital Segments: 10, 11, 12
        for pair in ("pt1", "pt2", "pt3"):  # Laser pairs: pt1, pt2, pt3
            # Attributes: longitude, latitude, h_corr, delta_time, etc
            corrected_height_ds = open_ATL11(
                atl11file=atl11file, group=f"{pair}/corrected_h"
            )
            reference_surface_ds = open_ATL11(
                atl11file=atl11file, group=f"{pair}/ref_surf"
            )
            ds = dask.delayed(obj=xr.combine_by_coords)(
                datasets=[corrected_height_ds, reference_surface_ds]
            )

            # Special exceptions to skip over
            if atl11file in (
                "ATL11.001/ATL11_014512_0103_03_v001.h5",
                "ATL11.001/ATL11_115810_0104_03_v001.h5",
            ) and pair in ("pt2", "pt3"):
                continue
                # print(atl11file, pair)
                # xr.open_dataset(
                #    atl11file, engine="h5netcdf", group=pair,
                # )  # will fail as is missing
            datasets.append(ds)

    dataset = dask.delayed(obj=xr.concat)(objs=datasets, dim="ref_pt")
    store_task = dataset.to_zarr(store=zarrfilepath, mode="w", consolidated=True)
    stores.append(store_task)

# %%
# Do all the HDF5 to Zarr conversion!
# Check conversion progress here, https://stackoverflow.com/a/37901797/6611055
futures = [client.compute(store_task) for store_task in stores]
for f in tqdm.tqdm(
    iterable=dask.distributed.as_completed(futures=futures), total=len(stores)
):
    pass

# %%
ds = xr.open_dataset(
    zarrfilepath, engine="zarr", backend_kwargs={"consolidated": True},
)
ds.h_corr.__array__().shape


# %% [raw]
# # Note, this raw conversion below takes about 11 hours
# # because HDF5 files work on a single thread...
# for atl11file in tqdm.tqdm(iterable=sorted(glob.glob("ATL11.001/*.h5"))):
#     name = os.path.basename(p=os.path.splitext(p=atl11file)[0])
#     zarr.convenience.copy_all(
#         source=h5py.File(name=atl11file, mode="r"),
#         dest=zarr.open_group(store=f"ATL11.001z/{name}.zarr", mode="w"),
#         if_exists="skip",
#         without_attrs=True,
#     )
