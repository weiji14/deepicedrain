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

# %%
# Adapted from https://github.com/suzanne64/ATL11/blob/master/intro_to_ATL11.ipynb
import os
import glob
import sys
import subprocess

import dask
import dask.distributed
import h5py
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
client = dask.distributed.Client(n_workers=64, threads_per_worker=1, processes=False)
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
# ## ATL06 to ATL11
#
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

# %%
# Note, conversion takes about 11 hours, because HDF5 files work on single thread...
for atl11file in tqdm.tqdm(iterable=sorted(glob.glob("ATL11.001/*.h5"))):
    name = os.path.basename(p=os.path.splitext(p=atl11file)[0])
    zarr.convenience.copy_all(
        source=h5py.File(name=atl11file, mode="r"),
        dest=zarr.open_group(store=f"ATL11.001z/{name}.zarr", mode="w"),
        if_exists="skip",
        without_attrs=True,
    )


# %%
sorted(os.listdir("ATL11.001/"))
thefile = "ATL11.001/ATL11_076211_0104_02_v001.h5"

with h5py.File(thefile, mode="r") as h5f:
    print(h5f.keys())
    print(h5f["pt1"].keys())

with xr.open_dataset(thefile, group="pt1/ref_surf", engine="h5netcdf") as rs:
    # read in the along-track coordinates
    x_atc = rs.x_atc.to_masked_array()
    # N slope
    n_slope = rs.n_slope.to_masked_array()
    e_slope = rs.e_slope.to_masked_array()
    bad = n_slope == 1.7976931348623157e308
    bad |= e_slope == 1.7976931348623157e308
    n_slope[bad] = np.NaN
    e_slope[bad] = np.NaN
    slope_mag = np.sqrt(n_slope ** 2 + e_slope ** 2)
    # get the reference-surface quality summary
    r_quality_summary = rs.quality_summary.to_masked_array()

with xr.open_dataset(thefile, group="pt1/corrected_h", engine="h5netcdf") as ch:
    # read in the along-track coordinates
    ref_pt = ch.ref_pt.to_masked_array()
    # read in the corrected_h
    h_corr = ch.h_corr.to_masked_array()
    # mask out invalid values
    bad = h_corr == 1.7976931348623157e308
    h_corr[bad] = np.NaN

    # error
    h_corr_sigma = ch.h_corr_sigma.to_masked_array()
    bad = h_corr_sigma == 1.7976931348623157e308
    h_corr_sigma[bad] = np.NaN
    # systematic error
    h_corr_sigma_s = ch.h_corr_sigma_systematic.to_masked_array()
    bad = h_corr_sigma == 1.7976931348623157e308
    h_corr_sigma_s[bad] = np.NaN
    # get the ATL06-based quality summary
    h_quality_summary = ch.quality_summary.to_masked_array()
    # read the cycle_number
    cycle_num = ch.cycle_number.to_masked_array()


if 1 == 1:
    plt.figure()
    plt.subplot(211)
    for cycle in range(0, h_corr.shape[1]):
        plt.plot(x_atc, h_corr[:, cycle], ".", label=f"cycle {cycle}")
    plt.legend()
    plt.xlabel("along-track distance")
    plt.ylabel("height, m")

    plt.subplot(212)
    plt.plot(x_atc, np.sum(np.isfinite(h_corr), axis=1), ".")
    plt.xlabel("along-track x")
    plt.ylabel("number of cycles present")

x_rep = np.tile(x_atc[:, np.newaxis], [1, len(cycle_num)])
q_rep = np.tile(r_quality_summary[:, np.newaxis], [1, len(cycle_num)])

if 2 == 2:
    plt.figure()
    plt.plot(x_atc, r_quality_summary, ".")
    plt.xlabel("x_atc")
    plt.ylabel("reference-surface quality summary")


if 3 == 3:
    fig = plt.figure(2)
    plt.clf()
    plt.plot(
        x_rep.ravel()[q_rep.ravel() != 6], h_corr.ravel()[q_rep.ravel() != 6], "k."
    )
    plt.title("points with surface quality not equal to 6")
    # plt.gca().set_xlim([6.75e6, 6.87e6])
    plt.tight_layout()
    plt.show()

if 4 == 4:
    comblist = list(itertools.combinations(cycle_num, 2))
    plt.figure(len(comblist) + 1, figsize=(4, 2 * len(comblist) + 1))
    plt.clf()
    ax = []
    good = np.flatnonzero(q_rep[:, 0] != 6)

    # cycle-to-cycle elevation differences
    for j, (col1, col2) in enumerate(comblist, start=1):
        ax += [plt.subplot(len(comblist) + 1, 1, j)]
        col1 -= 1
        col2 -= 1

        this_dh = h_corr[good, col2] - h_corr[good, col1]
        # cycle-to-cycle difference errors are the quadratic sums of the cycle errors
        this_dh_sigma = np.sqrt(
            h_corr_sigma[good, col2] ** 2 + h_corr_sigma[good, col1] ** 2
        )
        # Likewise for systematic errors:
        this_dh_sigma_s = np.sqrt(
            h_corr_sigma_s[good, col2] ** 2 + h_corr_sigma_s[good, col1] ** 2
        )
        plt.errorbar(
            x_rep[good, col2].ravel(),
            this_dh,
            yerr=np.sqrt(this_dh_sigma ** 2 + this_dh_sigma_s ** 2),
            fmt="r.",
        )
        plt.errorbar(x_rep[good, col2].ravel(), this_dh, yerr=this_dh_sigma, fmt="k.")
        ax[-1].set_ylabel(f"cycle {col2+1} \n minus \n cycle {col1+1}")
        ax[-1].set_ylim([-5, 5])

    # plot of the number of cycles available:
    ax += [plt.subplot(len(comblist) + 1, 1, j + 1)]
    plt.plot(x_atc, np.sum(np.isfinite(h_corr) & (q_rep != 6), axis=1), ".")
    ax[-1].set_ylabel("# of cycles available")
    ax[-1].set_ylim([0, 3.5])

# %% dh (change in elevation) over Antarctica!
def read_field(dataset: xr.Dataset, field: str):
    data = dataset[field].to_masked_array()
    bad1 = data == 1.7976931348623157e308
    data[bad1] = np.NaN
    bad2 = data == -1.7976931348623157e308
    data[bad2] = np.NaN
    return data


def read_ATL11(filepath: str, pair: str = "pt2", epsg: int = 3031):
    with xr.open_mfdataset(
        paths=filepath, group=f"{pair}/corrected_h", engine="h5netcdf"
    ) as ch:
        longitude = read_field(dataset=ch, field="longitude")
        latitude = read_field(dataset=ch, field="latitude")
        h_corr = read_field(dataset=ch, field="h_corr")
        h_corr_sigma = read_field(dataset=ch, field="h_corr_sigma")
        h_corr_sigma_s = read_field(dataset=ch, field="h_corr_sigma_systematic")
    with xr.open_mfdataset(
        paths=filepath, group=f"{pair}/ref_surf", engine="h5netcdf"
    ) as rs:
        x_atc = read_field(dataset=rs, field="x_atc")
        quality = rs["quality_summary"].to_masked_array().data
    h_corr[quality == 6] = np.NaN
    x, y = pyproj.Proj(projparams=epsg)(longitude, latitude)
    return x_atc, x, y, h_corr, np.sqrt(h_corr_sigma ** 2 + h_corr_sigma_s ** 2)


x_atc = []
x = []
y = []
h_corr = []
sigma_h = []
for pair in ["pt1", "pt2", "pt3"]:
    xx_atc, xx, yy, hh, ss = read_ATL11(
        filepath="ATL11.001/ATL11_????11_0105_02_v001.h5", pair=pair
    )
    x_atc += [xx_atc]
    x += [xx]
    y += [yy]
    h_corr += [hh]
    sigma_h += [ss]

# with xr.open_mfdataset(
#     paths="ATL11.001/ATL11_????11_0105_02_v001.h5",
#     group=f"{pair}/corrected_h",
#     engine="h5netcdf",
#     lock=False,
#     # combine="nested",
#     # concat_dim=None,
#     # drop_variables="delta_time"
# ) as ch:
#     pass

x_atc = np.concatenate(x_atc)
x = np.concatenate(x)
y = np.concatenate(y)
h_corr = np.concatenate(h_corr, axis=0)
sigma_h = np.concatenate(sigma_h, axis=0)

print(h_corr.shape)

if 5 == 5:
    c2: int = 5
    c1: int = 4
    plt.figure(figsize=[10, 10])
    plt.scatter(
        x=x[::20],
        y=y[::20],
        s=2,
        c=h_corr[::20, c2 - 1] - h_corr[::20, c1 - 1],
        vmin=-2,
        vmax=2,
        cmap="Spectral",
    )
    plt.axis("equal")
    hb = plt.colorbar()
    hb.set_label(f"cycle {c2} minus cycle {c1} elevation change (dh) in metres")

# TODO https://github.com/ICESAT-2HackWeek/elevation-change/blob/master/elevation_change_with_ATL11.ipynb

sdf = pd.DataFrame(sigma_h, columns=[f"s{i + 1}" for i in range(5)])

df = pd.concat(
    objs=[
        pd.DataFrame(data=x_atc, columns=["x_atc"]),
        pd.DataFrame(data=x, columns=["x"]),
        pd.DataFrame(data=y, columns=["y"]),
        pd.DataFrame(data=h_corr, columns=[f"h{i + 1}" for i in range(5)]),
    ],
    axis="columns",
)
df.head()

# TODO range of dh along window view of point with big change

# Cycle 1 - Spring2018 - 13Oct2018 - 28Dec2018  -ve MassBalance
# Cycle 2 - Summer2019 - 28Dec2018 - 29Mar2019 --ve MassBalance
# Cycle 3 - Autumn2019 - 29Mar2019 - 28Jun2019  +ve MassBalance *
# Cycle 4 - Winter2019 - 09Jul2019 - 26Sep2019 ++ve MassBalance *
# Cycle 5 - Spring2019 - 26Sep2019 - 26Dec2019  -ve MassBalance *
# Cycle 6 - Summer2020 - 26Dec2019 - 26Mar2020 --ve MassBalance

hmin = df[[f"h{i+1}" for i in range(5)]].min(axis="columns")  # minimum elevation
hmax = df[[f"h{i+1}" for i in range(5)]].max(axis="columns")  # maximum elevation
df["hrange"] = hmax - hmin  # range of elevation across all cycles
df.hrange.replace(to_replace=0.0, value=np.NaN, inplace=True)
df.to_csv("xyhr.csv")
# df = pd.read_csv("xyhr.csv", index_col=0)
bigdh = df[df["hrange"] > 5.5]  # find points where elevation range is greater than 5.5m
bigdh
bigdh.index

# TODO point in polygon (grounding line) to filter out ice shelf dynamics

for i in bigdh.index[:]:
    # i = 4848718
    temp_df = df.loc[i - 10 : i + 10]
    median_change = temp_df.hrange.median()
    if median_change >= 5.5 and median_change < 50:
        temp_sdf = sdf.loc[i - 10 : i + 10]
        for j in range(5):
            plt.errorbar(
                x=temp_df.x_atc,
                y=temp_df[f"h{j+1}"],
                yerr=temp_sdf[f"s{j+1}"],
                fmt="k.",
            )
            plt.scatter(x=temp_df.x_atc, y=temp_df[f"h{j+1}"], label=f"h{j+1}")
        plt.title(
            label=f"xy:{temp_df.loc[i].x},{temp_df.loc[i].y}\nindex:{i}, median_change:{median_change}m"
        )

        plt.gca().set_xlim(temp_df.x_atc[i - 10], temp_df.x_atc[i + 10])
        # plt.gca().set_ylim(160, 200)
        plt.legend()
        plt.show()

# Subglacial lake Slessor2 uplift
# -410918.8386,1029347.4666
# -408131.9125,1031128.9651
# Subglacial Lake Slessor4 drain
# -338117.9641,1110603.6373

# Subglacial Lake Whillans4/Mercer2 drainage
# -307154.8016,-507734.7378

# Subglacial Lake Macayeal 3 drainage (manually found)
# -734532.7023, -855436.2967

# Subglacial Lake Byrd 2 uplift (manually found, ~2m)
# 557187.1725,-855601.0561
# 555843.4189,-985710.3337 # Upstream Byrd Glacier ? rifting ??

# -741220.3139, 937483.8670 # Ronne-Filchner Ice Shelf
# -973351.7558, 272566.6157 # Ronne-Filchner Ice Shelf
# -1008445.1929,274272.3455  # Ronne-Filchner Ice Shelf
# 37261.1917,-1180880.8635 Ross Sea tidal motion
# -579964.1805,574791.5220 # Support Force Glacier at grounding line
# -1174079.4108,212533.0448 # Rutford Ice Stream/Shelf tidal motion?
