# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.11.3
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
import shutil
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

import deepicedrain

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# %%
client = dask.distributed.Client(n_workers=2, threads_per_worker=1)
client

# %% [markdown]
# ## Download ATL11 from [NSIDC](https://doi.org/10.5067/ATLAS/ATL11.003) up to cycle 9

# %%
# Note, need to downgrade using `pip install fsspec==0.7.4 intake-xarray==0.3.2`
# Get list of official ATL11 files to download
catalog = intake.open_catalog("deepicedrain/atlas_catalog.yaml")
with open(file="ATL11_to_download.txt", mode="r") as f:
    urlpaths = f.readlines()
dates: set = {url.split("/")[-2] for url in urlpaths}
len(dates)

# %%
# Submit download jobs to Client
futures = []
for date in dates:
    # date = "2019.11.01"  # sorted(dates)[-1]
    source = catalog.icesat2atl11(date=date)
    future = client.submit(
        func=source.discover, key=f"download-{date}"
    )  # triggers download of the file(s), or loads from cache
    futures.append(future)
    # break
    # source.urlpath

# %%
# Check download progress here, https://stackoverflow.com/a/37901797/6611055
responses = [f.result() for f in tqdm.tqdm(
    iterable=dask.distributed.as_completed(futures=futures), total=len(futures)
)]
# %%

# %% [markdown]
# ## Process ATL06 to ATL11 for cycle 9 or newer

# %%
# Create ATL06_to_ATL11 processing script, if not already present
if not os.path.exists("ATL06_to_ATL11_Antarctica.sh"):
    # find last cycle for each reference ground track and each orbital segment
    last_cycle_func = lambda rgt, ost: int(
        max(glob.glob(f"ATL06.003/**/ATL06*_*_{rgt:04d}??{ost:02d}_*.h5"))[-14:-12]
    )
    futures = []
    for referencegroundtrack, orbitalsegment in itertools.product(
        range(1387, 0, -1), [10, 11, 12]
    ):
        cyclenum = client.submit(
            last_cycle_func,
            referencegroundtrack,
            orbitalsegment,
            key=f"{referencegroundtrack:04d}-{orbitalsegment}",
        )
        futures.append(cyclenum)

    # Prepare string to write into ATL06_to_ATL11_Antarctica.sh bash script
    writelines = []
    for f in tqdm.tqdm(
        iterable=dask.distributed.as_completed(futures=futures), total=len(futures)
    ):
        referencegroundtrack, orbitalsegment = f.key.split("-")
        last_cycle = f.result()
        if last_cycle > 8:  # Only process those with Cycle 9 and newer locally
            writelines.append(
                f"python3 ATL11/ATL06_to_ATL11.py"
                f" {referencegroundtrack} {orbitalsegment}"
                f" --cycles 03 {last_cycle:02d}"
                f" --Release 2"
                f" --directory 'ATL06.003/**/'"
                f" --out_dir ATL11.002\n"
            )
            fname = f"ATL11_{referencegroundtrack}{orbitalsegment}_0308_002_01.h5"
            if not os.path.exists(f"ATL11.002/official/{fname}"):
                try:
                    shutil.move(src=f"ATL11.002/{fname}", dst="ATL11.002/official")
                except FileNotFoundError:
                    pass
        # else:  # Just use official NSIDC version for Cycle 8 or older
        # pass
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
# !PYTHONPATH=`pwd` PYTHONWARNINGS="ignore" parallel -a ATL06_to_ATL11_Antarctica.sh --bar --resume-failed --results logdir --joblog log --jobs 80 --load 100% > /dev/null

# %%
# df_log = pd.read_csv(filepath_or_buffer="log", sep="\t")
# df_log.query(expr="Exitval > 0")

# %% [markdown]
# ## Convert from HDF5 to Zarr format
#
# For faster data access speeds!
# We'll collect the data for each Reference Ground Track,
# and store it inside a Zarr format,
# specifically one that can be used by xarray.
# See also http://xarray.pydata.org/en/v0.15.1/io.html#zarr.
#
# Grouping hierarchy:
#   - Reference Ground Track (1-1387)
#     - Orbital Segments (10, 11, 12)
#       - Laser Pairs (pt1, pt2, pt3)
#         - Attributes (longitude, latitude, h_corr, delta_time, etc)

# %%
max_cycles: int = max(int(f[-11:-10]) for f in glob.glob("ATL11.003/*.h5"))
print(f"{max_cycles} ICESat-2 cycles available")

# %%
@dask.delayed
def open_ATL11(atl11file: str, group: str) -> xr.Dataset:
    """
    Opens up an ATL11 file using xarray and does some light pre-processing:
    - Mask values using _FillValue ??
    - Convert attribute format from binary to str
    """
    ds: xr.Dataset = xr.open_dataset(
        filename_or_obj=atl11file, group=group, engine="h5netcdf", mask_and_scale=True
    )

    # Change xarray.Dataset attributes from binary to str type
    # fixes issue when saving to Zarr format later
    # TypeError: Object of type bytes is not JSON serializable
    for key, variable in ds.variables.items():
        assert isinstance(ds[key].DIMENSION_LABELS, np.ndarray)
        ds[key].attrs["DIMENSION_LABELS"] = (
            ds[key].attrs["DIMENSION_LABELS"].astype(str)
        )
    try:
        ds.attrs["ATL06_xover_field_list"] = ds.attrs["ATL06_xover_field_list"].astype(
            str
        )
    except KeyError:
        pass

    return ds


# %%
@dask.delayed
def set_xy_and_mask(ds):
    # Calculate the EPSG:3031 x/y projection coordinates
    ds["x"], ds["y"] = deepicedrain.lonlat_to_xy(
        longitude=ds.longitude, latitude=ds.latitude
    )
    # Set x, y, x_atc and y_atc as coords of the xarray.Dataset instead of lon/lat
    ds: xr.Dataset = ds.set_coords(names=["x", "y", "x_atc", "y_atc"])
    ds: xr.Dataset = ds.reset_coords(names=["longitude", "latitude"])

    # Mask out low quality height data
    ds["h_corr"]: xr.DataArray = ds.h_corr.where(cond=ds.fit_quality == 0)

    return ds


# %%
# Consolidate together Antarctic orbital segments 10, 11, 12 into one file
# Also consolidate all three laser pairs pt1, pt2, pt3 into one file
atl11_dict = {}
for rgt in tqdm.trange(1387):
    atl11files: list = glob.glob(f"ATL11.003/ATL11_{rgt+1:04d}1?_????_00?_0?.h5")

    try:
        assert len(atl11files) == 3  # Should be 3 files for Orbital Segments 10,11,12
    except AssertionError:
        # Manually handle exceptional cases
        if len(atl11files) != 2:  # or rgt + 1 not in [1036]:
            raise ValueError(
                f"{rgt+1} only has {len(atl11files)} ATL11 files instead of 3"
            )

    if atl11files:
        pattern: dict = intake.source.utils.reverse_format(
            format_string="ATL11.003/ATL11_{referencegroundtrack:4}{orbitalsegment:2}_{cycles:4}_{version:3}_{revision:2}.h5",
            resolved_string=sorted(atl11files)[1],  # get the '11' one, not '10' or '12'
        )
        zarrfilepath: str = "ATL11.003z123/ATL11_{referencegroundtrack}1x_{cycles}_{version}_{revision}.zarr".format(
            **pattern
        )
        atl11_dict[zarrfilepath] = atl11files


# %%
# Get proper data encoding from a sample ATL11 file
atl11file: str = atl11files[0]
root_ds = open_ATL11(atl11file=atl11file, group="pt2").compute()
reference_surface_ds = open_ATL11(atl11file=atl11file, group="pt2/ref_surf").compute()
ds: xr.Dataset = xr.combine_by_coords(datasets=[root_ds, reference_surface_ds])

# Convert variables to correct datatype
encoding: dict = {}
df: pd.DataFrame = pd.read_csv("https://raw.githubusercontent.com/suzanne64/ATL11/master/ATL11/package_data/ATL11_output_attrs.csv")[["field", "datatype"]]
df = df.set_index("field")
for var in ds.variables:
    desired_dtype = str(df.datatype[var]).lower()
    if ds[var].dtype.name != desired_dtype:
        try:
            desired_dtype = desired_dtype.split(var)[1].strip()
        except IndexError:
            pass
    encoding[var] = {"dtype": desired_dtype}

# %%
# Gather up all the dask.delayed conversion tasks to store data into Zarr!
stores = []
for zarrfilepath, atl11files in tqdm.tqdm(iterable=atl11_dict.items()):
    zarr.open(store=zarrfilepath, mode="w")  # Make a new file/overwrite existing
    datasets = []
    for atl11file in atl11files:  # Orbital Segments: 10, 11, 12
        for pair in ("pt1", "pt2", "pt3"):  # Laser pairs: pt1, pt2, pt3
            # Attributes: longitude, latitude, h_corr, delta_time, etc
            root_ds = open_ATL11(atl11file=atl11file, group=pair)
            reference_surface_ds = open_ATL11(
                atl11file=atl11file, group=f"{pair}/ref_surf"
            )
            ds = dask.delayed(obj=xr.combine_by_coords)(
                datasets=[root_ds, reference_surface_ds]
            )
            # Light pre-processing
            ds = set_xy_and_mask(ds=ds)
            _rgt_array = dask.delayed(obj=np.full)(
                shape=ds.ref_pt.shape,
                fill_value=atl11file.split("_")[1][:4],
                dtype=np.int8,
            )
            ds = dask.delayed(obj=ds.assign_coords)(
                referencegroundtrack=("ref_pt", _rgt_array)
            )
            datasets.append(ds)

    dataset = dask.delayed(obj=xr.concat)(objs=datasets, dim="ref_pt")
    store_task = dataset.to_zarr(
        store=zarrfilepath, mode="w", encoding=encoding, consolidated=True
    )
    stores.append(store_task)

# %%
# Do all the HDF5 to Zarr conversion! Should take about half an hour to run
# Check conversion progress here, https://stackoverflow.com/a/37901797/6611055
futures = [client.compute(store_task) for store_task in stores]
for _ in tqdm.tqdm(
    iterable=dask.distributed.as_completed(futures=futures), total=len(stores)
):
    pass

# %%
ds = xr.open_dataset(zarrfilepath, engine="zarr", backend_kwargs={"consolidated": True})
ds.h_corr.__array__().shape


# %% [raw]
# # Note, this raw conversion below takes about 11 hours
# # because HDF5 files work on a single thread...
# for atl11file in tqdm.tqdm(iterable=sorted(glob.glob("ATL11.003/*.h5"))):
#     name = os.path.basename(p=os.path.splitext(p=atl11file)[0])
#     zarr.convenience.copy_all(
#         source=h5py.File(name=atl11file, mode="r"),
#         dest=zarr.open_group(store=f"ATL11.003z/{name}.zarr", mode="w"),
#         if_exists="skip",
#         without_attrs=True,
#     )
