# %% [markdown]
# # Single Baseline Filtering and Power Spectrum Estimation
# 
# **by Josh Dillon, Bobby Pascua, Mike Wilensky, Jianrong Tan, Steven Murray, and Tyler Cox**, last updated March 16, 2026
# 
# This notebook is designed to take a single redundantly-averaged unique baseline (typically after LST-binning) and push it through all the way to the power spectrum. It operates on single files that contain a single baseline for all LSTs and both `'ee'` and `'nn'` polarizations. It then can:
# * Throw out highly flagged times and/or channels
# * Inpaint autocorrelations to produce a noise model, if necessary
# * Inpaint cross-correlations (optional)
# * Delay-filter cross-correlations (optional)
# * De-interleave by time into multiple waterfalls with independent noise and rephased to the same set of LSTs
# * Perform crosstalk notch filtering of the FR = 0 mode
# * Perform main beam top hat fringe-rate filtering
# * Convert to pseudo-Stokes I and Q
# * Perform coherent time averaging
# * Compute power spectra from pairs of interleaves
# * Estimate noise, accounting for how the fringe-rate filter and the coherent 
# * Incoherent averaging over time and across interleave-pairs
# 
# This notebook also produces a series of plots and tables to illustrate the progress of the analysis. These include:
# 
# ### [• Table 1: Band Definitions](#Table-1:-Band-Definitions)
# ### [• Figure 1: Bands and Flag Occupancy](#Figure-1:-Bands-and-Flag-Occupancy)
# ### [• Table 2: Fringe-Rate and Crosstalk Filtering Ranges and Losses](#Table-2:-Fringe-Rate-and-Crosstalk-Filtering-Ranges-and-Losses)
# ### [• Figure 2: Waterfalls Before Delay Filtering and/or Inpainting](#Figure-2:-Waterfalls-Before-Delay-Filtering-and/or-Inpainting)
# ### [• Figure 3: Waterfalls After Delay Filtering and/or Inpainting](#Figure-3:-Waterfalls-After-Delay-Filtering-and/or-Inpainting)
# ### [• Figure 4: First Set of De-Interleaved Waterfalls after Cross-Talk Filtering](#Figure-4:-First-Set-of-De-Interleaved-Waterfalls-after-Cross-Talk-Filtering)
# ### [• Figure 5: First Set of De-Interleaved Waterfalls after Main-Beam Fringe-Rate Filtering](#Figure-5:-First-Set-of-De-Interleaved-Waterfalls-after-Main-Beam-Fringe-Rate-Filtering)
# ### [• Figure 6: First Set of De-Interleaved Waterfalls after Coherent Time Averaging](#Figure-6:-First-Set-of-De-Interleaved-Waterfalls-after-Coherent-Time-Averaging)
# ### [• Figure 7: First Set of De-Interleaved Waterfalls after Forming Pseudo-Stokes I](#Figure-7:-First-Set-of-De-Interleaved-Waterfalls-after-Forming-Pseudo-Stokes-I)
# ### [• Figure 8: Interleave-Averaged Power Spectra (Pseudo-Stokes I, Q, U, & V) vs. LST](#Figure-8:-Interleave-Averaged-Power-Spectra-(Pseudo-Stokes-I,-Q,-U,-&-V)-vs.-LST)
# ### [• Figure 9: Interleave-Averaged Power Spectrum SNR vs. LST (Real and Imaginary for pI)](#Figure-9:-Interleave-Averaged-Power-Spectrum-SNR-vs.-LST-(Real-and-Imaginary-for-pI))
# ### [• Figure 10: High Delay Power Spectrum SNR Histograms Before and After Incoherent Averaging](#Figure-10:-High-Delay-Power-Spectrum-SNR-Histograms-Before-and-After-Incoherent-Averaging)
# ### [• Figure 11: Incoherently Averaged Power Spectrum with Error Bars](#Figure-11:-Incoherently-Averaged-Power-Spectrum-with-Error-Bars)
# 

# %% [markdown]
# ## Imports and Parameters

# %% [markdown]
# **TODO:**
# 
# * Calculate signal loss due to redundant averaging

# %%
import time
tstart = time.time()

# %%
import os
import sys
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import h5py
import hdf5plugin  # REQUIRED to have the compression plugins available
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib
import copy
import warnings
from astropy import units
from scipy import constants, interpolate, special
from scipy.signal.windows import dpss
from pyuvdata import UVBeam
from pyuvdata import utils as uvutils
from hera_cal import io, utils, vis_clean, frf, datacontainer, noise, redcal
from hera_qm.time_series_metrics import true_stretches
from hera_filters import dspec
import hera_pspec as hp
import uvtools
from functools import reduce
from IPython.display import HTML
from pathlib import Path
from hera_notebook_templates.utils import parse_band_str
import importlib

# %%
import toml

#Global tracker vars
is_delay_analysis = False

# Settings configurable via env vars (typically set by the bash wrapper).
SINGLE_BL_FILE: str
OUT_BL_DELAY_RANGE_FILE: str
OUT_PSPEC_FILE: str
OUT_TAVG_PSPEC_FILE: str
TOML_FILE = os.environ.get('TOML_FILE',
        '/home/Kwuzard/Projects/HERA_HONS/src/HONS/h6c_pspec_11band.toml')

# Load configuration from the toml and inject into globals.
toml_options = toml.load(TOML_FILE)
for toml_section in ['GLOBAL_OPTS', 'POSTPROCESS_AND_PSPEC_OPTS']:
    if toml_section not in toml_options:
        continue
    print(f'\nLoading config from [{toml_section}] in {TOML_FILE}:')
    for key, val in toml_options[toml_section].items():
        globals()[key.upper()] = val
        #print(f'  {key.upper()} = {val!r}')


# Settings configurable via env vars (typically set by the bash wrapper). Continued.
if __name__ == "__main__" and len(sys.argv) == 6:
    # Means the file that imported this file is giving us the single bl file to work with.
    input_file = Path(sys.argv[1])
    output_dir = Path(sys.argv[3])

    PERFORM_DLY_FILT = sys.argv[2].lower() == "true"
    OUT_PSPEC_FILE = output_dir / f"{input_file.stem}.pspec.h5"
    OUT_TAVG_PSPEC_FILE = PERFORM_DLY_FILT and output_dir / f"{input_file.stem}_cutoff_{sys.argv[4]}.tavg.pspec.h5" or (
        output_dir / f"{input_file.stem}.tavg.pspec.h5")
    OUT_BL_DELAY_RANGE_FILE = output_dir / f"{input_file.stem}_cutoff_{sys.argv[4]}.tavg.delay_filter_hw.csv"
    SINGLE_BL_FILE = str(input_file)
    DLY_FILT_MIN_DLY = int(sys.argv[4])
    PERFORM_INPAINT = sys.argv[5].lower() == "true"

    is_delay_analysis = True
    print("Override default params.")
else:
    SINGLE_BL_FILE = os.environ.get('SINGLE_BL_FILE',
        '/home/Kwuzard/Projects/HERA_HONS/raw_data/single_baselines_raw_data/zen.LST.baseline.0_2.sum.FR0filt.uvh5')
    OUT_PSPEC_FILE = os.environ.get('OUT_PSPEC_FILE',
        str(Path(SINGLE_BL_FILE).with_suffix('.pspec.h5')))
    OUT_TAVG_PSPEC_FILE = os.environ.get('OUT_TAVG_PSPEC_FILE',
        str(Path(SINGLE_BL_FILE).with_suffix('.tavg.pspec.h5')))

for setting in ['TOML_FILE', 'SINGLE_BL_FILE', 'OUT_PSPEC_FILE', 'OUT_TAVG_PSPEC_FILE']:
    print(f'{setting} = "{eval(setting)}"')

# Advanced / debugging knobs that aren't in the toml — keep their defaults.
FLAG_COHERENT_CHUNKS: bool = False
USE_CORR_MATRIX: bool = True
CORR_MATRIX_FREQ_DECIMATION: int = 10
CORR_MATRIX_NOTCH_CUTOFF: float = 30.0
USE_SIMULATED_NOISE: bool = False
FLAT_AUTOS: bool = False
NO_FLAGS_FLAT_NSAMPLES: bool = False
SKIP_XTALK_AND_FRF: bool = False

# %%
if (not PERFORM_INPAINT) and (not PERFORM_DLY_FILT):
    CHANNEL_FLAG_CUT = 0.0
    PIXEL_FLAG_CUT = 0.0

# %%
SINGLE_BL_FILE = Path(SINGLE_BL_FILE)
FR_SPECTRA_FILE = Path(FR_SPECTRA_FILE)
EFIELD_HEALPIX_BEAM_FILE = Path(EFIELD_HEALPIX_BEAM_FILE)

# %%
if SAVE_RESULTS and OUT_PSPEC_FILE is None:
    OUT_PSPEC_FILE = SINGLE_BL_FILE.with_suffix(f".pspec.h5")
if SAVE_RESULTS and OUT_TAVG_PSPEC_FILE is None:
    OUT_TAVG_PSPEC_FILE = SINGLE_BL_FILE.with_suffix(f".tavg.pspec.h5")

# %% [markdown]
# ## Load Data

# %%
# figure out ANTPAIR and corresponding AUTO_BL_FILE
ANTPAIR = tuple([int(ant) for ant in re.search(r'\d+_\d+', SINGLE_BL_FILE.name).group().split('_')])
all_files = SINGLE_BL_FILE.parent.glob(SINGLE_BL_FILE.name.replace(f'{ANTPAIR[0]}_{ANTPAIR[1]}', '*'))
AUTO_BL_FILE = sorted([f for f in all_files if len(set(re.search(r'\d+_\d+', f.name).group().split('_'))) == 1])[0]

# %%
# load data for both crosses and autos with times corresponding only to those in the crosses,
single_bl_times = np.array(io.HERAData(SINGLE_BL_FILE).times)
hd = io.HERAData([SINGLE_BL_FILE, AUTO_BL_FILE])
print(SINGLE_BL_FILE)
print(AUTO_BL_FILE)
data, flags, nsamples = hd.read(times=single_bl_times)
cross_bls = [ANTPAIR + (pol,) for pol in data.pols()]

# %%
# check that non-finite data is flagged and flagged data is set to 0
for bl in cross_bls:
    assert np.all(flags[bl][~np.isfinite(data[bl])])
    data[bl][~np.isfinite(data[bl])] = 0
    data[bl][flags[bl]] = 0

# %%
# Wrap-aware LST helpers (BAND_STR-style hour ranges -> radian masks).
def parse_lst_ranges_rad(ranges_str):
    """Parse comma-separated `lo~hi` hour ranges and return a list of (lo_rad, hi_rad)
    pieces, each with lo_rad <= hi_rad in [0, 2*pi]. Wrapping ranges (lo > hi after mod 24)
    are split into two non-wrap pieces straddling 0."""
    out = []
    for r in ranges_str.strip().split(','):
        lo_h, hi_h = (float(e) for e in r.split('~'))
        lo_r = (lo_h % 24) * np.pi / 12
        # Treat a positive multiple of 24 as the full-circle upper bound, not 0.
        hi_r = 2 * np.pi if (hi_h != 0 and hi_h % 24 == 0) else (hi_h % 24) * np.pi / 12
        if lo_r <= hi_r:
            out.append((lo_r, hi_r))
        else:
            out.extend([(lo_r, 2 * np.pi), (0.0, hi_r)])
    return out

def lst_in_ranges(lsts_rad, ranges_rad):
    mask = np.zeros_like(lsts_rad, dtype=bool)
    for lo_r, hi_r in ranges_rad:
        mask |= (lsts_rad >= lo_r) & (lsts_rad <= hi_r)
    return mask

# Flag integrations with LSTs outside [LST_MIN, LST_MAX] (hours).
lst_outside = ~lst_in_ranges(data.lsts, parse_lst_ranges_rad(f"{LST_MIN}~{LST_MAX}"))
for bl in flags:
    flags[bl][lst_outside, :] = True
for bl in cross_bls:
    data[bl][flags[bl]] = 0

# %%
auto_antpair = sorted(set([k[0:2] for k in data.bls() if k[0] == k[1]]))[0]
df = np.median(np.diff(data.freqs))
dt = np.median(np.diff(data.times)) * 24 * 3600
# Calculate averaging time that divides neatly into NINTERLEAVE
AVERAGING_TIME = TARGET_AVERAGING_TIME / (dt * (1 + 1e-10)) // NINTERLEAVE * (dt * (1 + 1e-10)) * NINTERLEAVE 
print(f'Using an actual coherent averaging time of {AVERAGING_TIME:.3f} seconds to ensure even interleaving.')

# %% [markdown]
# ## NSamples Cut

# %%
# Removes all pixels that are more than 50% (by default) flagged relative to the maximum in that integration.
# This is done above and below FM separately because it's possible for a baseline to be entirely flagged above or below but not both.
FM_ind = np.argmin(np.abs(data.freqs - FM_CUT_FREQ))
for bl in cross_bls:
    npix_flagged_before = np.sum(nsamples[bl] == 0)
    for fslice in [slice(0, FM_ind), slice(FM_ind, len(data.freqs))]:
        if fslice.start >= fslice.stop:
            continue
        flags[bl][:, fslice][(nsamples[bl][:, fslice] < PIXEL_FLAG_CUT * np.max(nsamples[bl][:, fslice], axis=1, keepdims=True))] = True
        nsamples[bl][flags[bl]] = 0
    print(f'{bl}: flagging {np.sum(nsamples[bl] == 0) - npix_flagged_before} pixels.')    

# Remove all integrations that have fewer integrations than 20% (by default) of the best-observed integration
nsamples_by_time = np.sum([np.where(flags[bl], 0, nsamples[bl]) for bl in cross_bls], axis=(0, 2))
for bl in cross_bls:
    print(f'{bl}: flagging {np.sum((nsamples_by_time < INTEGRATION_FLAG_CUT  * np.max(nsamples_by_time)) & ~np.all(flags[bl], axis=1))} times.')
    flags[bl][nsamples_by_time < INTEGRATION_FLAG_CUT  * np.max(nsamples_by_time), :] = True
    nsamples[bl][flags[bl]] = 0

# Remove all channels that are more than 50% (by default) flagged (relative to the best-observed channel)
nsamples_by_chan = np.sum([np.where(flags[bl], 0, nsamples[bl]) for bl in cross_bls], axis=(0, 1))
for bl in cross_bls:
    print(f'{bl}: flagging {np.sum((nsamples_by_chan < CHANNEL_FLAG_CUT * np.max(nsamples_by_chan)) & ~np.all(flags[bl], axis=0))} channels.')    
    flags[bl][:, nsamples_by_chan < CHANNEL_FLAG_CUT * np.max(nsamples_by_chan)] = True
    nsamples[bl][flags[bl]] = 0

# %% [markdown]
# ## Define and show bands

# %%
df, bands, min_freqs, max_freqs, band_slices, nchans = parse_band_str(BAND_STR, data.freqs)
min_chan = [slc.start for slc in band_slices]
max_chan = [slc.stop for slc in band_slices]

all_zs = 1420405751.768 / data.freqs - 1
avg_zs = [np.mean(all_zs[slc]) for slc in band_slices]

bandwidth = [f'{nc * df / 1e6:.1f}' for nc in nchans]

# %%
# If inpainting as already been done, but there are unflagged channels in any of the pre-defined power spectrum bands, flag the whole band.
if ALREADY_INPAINTED:
    for i, band_slice in enumerate(band_slices):
        for bl in cross_bls:
            # find integrations that are not entirely flagged (these don't count when it comes to missing channels)
            flagged_ints = np.all(flags[bl][:, band_slice], axis=1)
            if np.any(flags[bl][~flagged_ints, band_slice]):
                print(f'{bl}: flagging Band {i+1} for not being completely inpainted.')
                flags[bl][:, band_slice] = True
                nsamples[bl][flags[bl]] = 0

# %%
def plot_bands():
    plt.figure(figsize=(18, 6), dpi=100)
    to_plot = np.mean([nsamples[ANTPAIR + ('ee',)], nsamples[ANTPAIR + ('nn',)]], axis=(0, 1)) 
    plt.plot(data.freqs/1e6, to_plot, 'k.-', lw=.5, ms=4, label='Included in a band')

    plt.xlabel('Frequency (MHz)')
    plt.ylabel(f'Average Nsamples on {ANTPAIR}')

    in_any_band = np.sum([(data.freqs / 1e6 <= band[1]) & (data.freqs / 1e6 >= band[0]) for band in bands], axis=0).astype(bool)
    plt.plot(data.freqs[~in_any_band] / 1e6, to_plot[~in_any_band], 'r.', lw=.5, ms=4, label='Excluded from all bands')
    plt.legend(loc='lower left')

    for i, band in enumerate(bands):
        plt.axvspan(band[0], band[1], alpha=.3, color=f'C{i}', zorder=0)
        plt.text((band[0] + band[1]) / 2, np.max(to_plot) * 1.05, f'Band {i + 1}', ha='center', va='bottom',
                 bbox=dict(facecolor='w', edgecolor='black', alpha=.75, boxstyle='round'))

    plt.ylim([-1, np.max(to_plot) * 1.1])    

    for freq in [117.19, 133.11, 152.25, 167.97]:
        plt.axvline(freq, ls='--', color='k')

    for i, freq in enumerate([(117.19 + 133.11)/2, (152.25 + 167.97)/2]):    
        plt.text(freq, np.max(to_plot) * .03, f'H1C IDR3\nBand {i + 1}', ha='center', va='bottom',
                 bbox=dict(facecolor='w', edgecolor='black', alpha=.75, boxstyle='round', ls='--'))

    plt.tight_layout()

def show_band_table():
    table = pd.DataFrame({'Band': np.arange(len(bands)) + 1,
                          'Channel Range': [f'{c0} — {c1}' for c0, c1 in zip(min_chan, max_chan)],
                          '# of Channels': nchans,
                          'Frequency Range (MHz)': [f'{f0:.1f} — {f1:.1f}' for f0, f1 in zip(min_freqs, max_freqs)],
                          '$\\Delta\\nu$ (MHz)': bandwidth,
                          '$z$ Range': [f'{1420.405751768 / f1 - 1:.2f} — {1420.405751768 / f0 - 1:.1f}' for f0, f1 in zip(min_freqs, max_freqs)],
                          'Center $z$': [f'{(1420.405751768 / f1 - 1) / 2 +  (1420.405751768 / f0 - 1) / 2:.1f}' for f0, f1 in zip(min_freqs, max_freqs)],
                          'Delta $z$': [f'{(1420.405751768 / f0 - 1) -  (1420.405751768 / f1 - 1):.1f}' for f0, f1 in zip(min_freqs, max_freqs)],
                         })
    return table.style.hide().to_html()

# %% [markdown]
# ### *Table 1: Band Definitions*

# %%
HTML(show_band_table())

# %% [markdown]
# ### *Figure 1: Bands and Flag Occupancy*
# 
# This figure illustrates the definition of the various bands in which the power spectrum is to be estimated, which frequencies are included/excluded, as well as the showing the fraction of each channel flagged.

# %%
if PLOT: plot_bands()

# %% [markdown]
# ## Figure out slicing

# %%
# figure out high and low bands
FM_ind = np.argmin(np.abs(data.freqs - FM_CUT_FREQ))
unflagged_chans = np.argwhere(~np.all([np.all(flags[bl], axis=0) for bl in flags], axis=0)).squeeze()
if np.any(unflagged_chans < FM_ind):
    low_band = slice(np.min(unflagged_chans), np.max(unflagged_chans[unflagged_chans < FM_ind]) + 1)
else:
    low_band = slice(0,0)
if np.any(unflagged_chans > FM_ind):
    high_band = slice(np.min(unflagged_chans[unflagged_chans > FM_ind]), np.max(unflagged_chans) + 1)
else:
    high_band = slice(0,0)
    
print(f'Below FM Frequency Slice: {low_band}')
print(f'Above FM Frequency Slice: {high_band}')

# figure out the range of times that includes all unflagged times (though may still have some flags) for all polarizations
ORed_flags = np.any([np.all(flags[bl], axis=1) for bl in cross_bls], axis=0)
tslice = slice(true_stretches(~ORed_flags)[0].start, true_stretches(~ORed_flags)[-1].stop)
print(f'Time Slice Excluded Edge Flags: {tslice}')

# %%
# FOR DIAGNOSTICS/DEBUGGING ONLY: unflag everything and set nsamples to the median
if NO_FLAGS_FLAT_NSAMPLES:
    for bl in cross_bls:
        flags[bl][tslice, :] = False
        nsamples[bl][tslice, :] = np.median([nsamples[bl][tslice, :] for bl in cross_bls])
    
# FOR DIAGNOSTICS/DEBUGGING ONLY: make all the autos a flat 10,000 Jy 
if FLAT_AUTOS:
    for bl in data:
        if bl[0] == bl[1]:
            data[bl] = 10000 * np.ones_like(data[bl])

# %% [markdown]
# ## Filtering and Systematics Mitigation

# %% [markdown]
# ### Plotting Functions

# %%
def sym_log_norm(to_plot, linthresh=10, clim=None):
    '''Convenience interface for matplotlib.colors.SymLogNorm'''
    if clim is None:
        return matplotlib.colors.SymLogNorm(linthresh, vmin=-np.nanmax(np.abs(to_plot)), vmax=np.nanmax(np.abs(to_plot)))
    else:
        return matplotlib.colors.SymLogNorm(linthresh, vmin=clim[0], vmax=clim[1])

# %%
def plot_waterfall(data, bl=(ANTPAIR + ('ee',)), flags=flags, nsamples=nsamples, tslice=tslice):
    '''Plots data (amplitude and phase) as well as nsamples waterfalls for a baseline.'''
    if tslice is None:
        tslice = slice(0, data[bl].shape[0], 1)
    lsts = np.where(data.lsts > data.lsts[-1], data.lsts - 2 * np.pi, data.lsts)[tslice] * 12 / np.pi
    extent = [data.freqs[0]/1e6, data.freqs[-1]/1e6, lsts[-1], lsts[0]]
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 12), sharex=True, sharey=True, dpi=100)
    im = axes[0].imshow(np.where(flags[bl], np.nan, np.abs(data[bl]))[tslice], aspect='auto', norm=matplotlib.colors.LogNorm(), interpolation='none', cmap='inferno', extent=extent)
    fig.colorbar(im, ax=axes[0], location='top', pad=.02, label=f'{bl}: Amplitude (Jy)')

    im = axes[1].imshow(np.where(flags[bl], np.nan, np.angle(data[bl]))[tslice], aspect='auto', cmap='twilight', interpolation='none', extent=extent)
    fig.colorbar(im, ax=axes[1], location='top', pad=.02, label=f'{bl}: Phase (Radians)')

    im = axes[2].imshow(nsamples[bl][tslice], aspect='auto', interpolation='none', extent=extent)
    fig.colorbar(im, ax=axes[2], location='top', pad=.02, label=f'{bl}: Number of Samples')
    plt.tight_layout()

    for ax in axes:
        ax.set_ylabel('LST (Hours)')
        ax.set_xlabel('Frequency (MHz)')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ax.set_yticklabels([f'{(int(val) if np.isclose(val, int(val)) else val) % 24:n}' for val in ax.get_yticks()])

    plt.tight_layout()

# %%
def plot_real_delay_vs_lst(data, bl=(ANTPAIR + ('ee',)), flags=None, xlim=[-1999, 1999], clim=None, linthresh=10, taper=TAPER, tslice=tslice):
    '''Plots the real part of the tapered FFT of the data in each power spectrum band as a function of delay and LST.'''
    if tslice is None:
        tslice = slice(0, data[bl].shape[0], 1)
    lsts = np.where(data.lsts > data.lsts[-1], data.lsts - 2 * np.pi, data.lsts)[tslice] * 12 / np.pi

    fig, axes = plt.subplots(1, len(bands), figsize=(28, 12), sharex=True, sharey=True, gridspec_kw={'wspace': .03}, dpi=100)
    for i, (ax, band, band_slice) in enumerate(zip(axes, bands, band_slices)):
        dfft = uvtools.utils.FFT(data[bl][tslice, band_slice], axis=1, taper=taper)
        delays = uvtools.utils.fourier_freqs(data.freqs[band_slice]) * 1e9
        to_plot = np.real(dfft)
        if flags is not None:
            flagged_times = np.all(flags[bl][tslice, band_slice], axis=1)
            to_plot[flagged_times, :] = np.nan
        if i == 0:
            _to_plot = copy.deepcopy(to_plot)
            ax.set_ylabel('LST (Hours)')
        im = ax.imshow(to_plot, interpolation='none', aspect='auto', cmap='bwr', norm=sym_log_norm(_to_plot, linthresh=linthresh, clim=clim),
                       extent=[delays[0], delays[-1], lsts[-1], lsts[0]])
        for dly in dly_filter_half_widths[0] * 1e9 * np.array([1, -1]):
            ax.axvline(dly, ls='--', color='k', lw=.5)
        for dly in inpaint_filter_half_widths[0] * 1e9 * np.array([1, -1]):
            ax.axvline(dly, ls=':', color='k', lw=.5)
        ax.set_xlim(xlim)
        ax.set_title(f'Band {i+1}:\n{band[0]}—{band[1]} MHz', fontsize=10)
        ax.set_xlabel('Delay (ns)')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ax.set_yticklabels([f'{(int(val) if np.isclose(val, int(val)) else val) % 24:n}' for val in ax.get_yticks()])

    plt.colorbar(im, ax=axes, pad=.02, aspect=40, extend='both', label=f'Re$[\\widetilde{{V}}_{{{bl}}}]$ (Jy)')

# %%
def plot_dly_vs_fr(data, bl=(ANTPAIR + ('ee',)), xlim=[-1999, 1999], ylim=[-5, 5], clim=None, tslice=tslice, taper=TAPER):
    '''Plots the magnitude of the 2D tapered FFT of the data in each power spectrum band as a function of delay and FR.
    Also shows the foreground filtering delay, the main beam range of FRs, and the expected shape of mutual coupling.'''
    fig, axes = plt.subplots(1, len(bands), figsize=(28, 6), sharex=True, sharey=True, gridspec_kw={'wspace': .03}, dpi=100)
    if tslice is None:
        tslice = slice(0, data[bl].shape[0], 1)

    to_plots = []
    for i, (ax, band, band_slice) in enumerate(zip(axes, bands, band_slices)):
        dfft = uvtools.utils.FFT(data[bl][:, band_slice], axis=1, taper=TAPER)
        delays = uvtools.utils.fourier_freqs(data.freqs[band_slice]) * 1e9
        frates = uvtools.utils.fourier_freqs((data.times[tslice] - data.times[0]) * 24 * 60 * 60) * 1000

        dfft2 = uvtools.utils.FFT(dfft[tslice, :], axis=0, taper=TAPER)

        to_plot = np.abs(dfft2)
        to_plots.append(to_plot)
        if i == 0:
            _to_plot = copy.deepcopy(to_plot)
            ax.set_ylabel('Fringe Rate (mHz)')        
        im = ax.imshow(to_plot, interpolation='none', aspect='auto', cmap='turbo', 
                       norm=matplotlib.colors.LogNorm(vmin=(clim[0] if clim is not None else np.min(_to_plot)), 
                                                      vmax=(clim[1] if clim is not None else np.max(_to_plot))),
                       extent=[delays[0], delays[-1], frates[-1], frates[0]])
        for dly in dly_filter_half_widths[0] * 1e9 * np.array([1, -1]):
            ax.axvline(dly, ls='--', color='k', lw=.5)
        for dly in inpaint_filter_half_widths[0] * 1e9 * np.array([1, -1]):
            ax.axvline(dly, ls=':', color='k', lw=.5)
        for fr in fr_ranges[band]:# + (pol,)]:
            ax.axhline(fr, ls='--', color='k', lw=.5)

        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.set_title(f'Band {i+1}:\n{band[0]}—{band[1]} MHz', fontsize=10)
        ax.set_xlabel('Delay (ns)')

    plt.colorbar(im, ax=axes, pad=.02, aspect=40, extend='both', label=f'$|\\widetilde{{V}}_{{{bl}}}|$ (Jy)')
    
    omega_earth = 2 * np.pi / (24 * 3600) #rad/s
    hera_dec = -30.72152612068925 * np.pi / 180
    def calculate_fr(bl_vec, freq):
        """bl_vec in meters, freq in Hz, returns fr in mHz"""
        bl_we = bl_vec[0]
        fr = -bl_we * omega_earth * freq * np.cos(hera_dec) / constants.c * 1e3
        return fr
    def max_fr(tau, freq):
        """takes in tau in ns, freq in Hz, returns corresponding max fringe rate in mHz"""
        return omega_earth * freq * tau * np.cos(hera_dec) * 1e-6 # mHz 

    b_ij = data.antpos[bl[1]] - data.antpos[bl[0]]
    for i in range(len(bands)):
        freq = np.mean(bands[i]) * 1e6
        axes[i].plot(delays, max_fr(delays, freq) + calculate_fr(b_ij, freq), color='k', ls='--', lw=.5)
        axes[i].plot(delays, -max_fr(delays, freq) + calculate_fr(b_ij, freq), color='k', ls='--', lw=.5)

# %% [markdown]
# ### Filtering and Post-Processing Functions

# %%
def delay_filter(data, wgts, filter_centers, filter_half_widths, eigenval_cutoff, cache={}, bls=None, zeros_where_zero_wgt=True):
    '''This function performs a high-pass delay filter, removing the wedge plus some buffer. It also performs inpainting with the same delay.'''
    dly_filt_data = copy.deepcopy(data)
    inpainted_data = copy.deepcopy(data)
    if bls is None:
        bls = cross_bls
    
    for bl in bls:
        d_mdl = np.zeros_like(dly_filt_data[bl])
        for band in [low_band, high_band]:
            if band.start >= band.stop:
                # This can happen if the frequencies are all above/below FM
                continue
            d_mdl[:, band], _, info = dspec.fourier_filter(data.freqs[band], dly_filt_data[bl][:, band], wgts=wgts[bl][:, band], filter_centers=filter_centers, 
                                                           filter_half_widths=filter_half_widths, mode='dpss_solve', 
                                                           eigenval_cutoff=[eigenval_cutoff], suppression_factors=[eigenval_cutoff], 
                                                           max_contiguous_edge_flags=len(data.freqs), cache=cache)
        if zeros_where_zero_wgt:
            dly_filt_data[bl] = np.where(wgts[bl] == 0, 0, dly_filt_data[bl] - d_mdl)
        else:
            dly_filt_data[bl] = dly_filt_data[bl] - d_mdl
        inpainted_data[bl] = np.where(wgts[bl] == 0, d_mdl, data[bl])
        #d_mdl is modeled. d_mdl is an array over delays. Thus, subtracting d_mdl from dly_filt_data is like subtracting actual data in delay space from the delay model.
    
    return dly_filt_data, inpainted_data

# %%
def xtalk_filter(data, wgts, xtalk_fr=XTALK_FR, tslice=tslice, cache={}, bls=None):
    '''This function performs a high-pass filter in fringe rate, removing some small range around the 0 FR mode.'''
    xtalk_filt_data = copy.deepcopy(data)
    if bls is None:
        bls = cross_bls
    
    for bl in bls:
        if tslice is None:
            tslice = slice(0, data[bl].shape[0], 1)

        d_mdl, _, info = dspec.fourier_filter(data.times[tslice] * 24 * 60 * 60, data[bl][tslice], 
                                              wgts=wgts[bl][tslice, :], filter_centers=[0], 
                                              filter_half_widths=[xtalk_fr / 1000], mode='dpss_solve', 
                                              eigenval_cutoff=[FR_EIGENVAL_CUTOFF], suppression_factors=[FR_EIGENVAL_CUTOFF], 
                                              max_contiguous_edge_flags=len(data.times), cache=cache, filter_dims=0)

        out = xtalk_filt_data[bl].copy()
        out[tslice, :] = np.where(wgts[bl][tslice, :] == 0, 0, xtalk_filt_data[bl][tslice, :] - d_mdl)
        xtalk_filt_data[bl] = out
    return xtalk_filt_data

# %%
def main_beam_FR_filter(data, wgts, tslice=tslice, cache={}, bls=None):
    '''This function performs fringe rate filtering, keeping a range determined by fr_ranges for each band.'''
    frf_data = copy.deepcopy(data)
    if bls is None:
        bls = cross_bls
    
    info = {}
    for bl in bls:    
        if tslice is None:
            tslice = slice(0, data[bl].shape[0], 1)

        info[bl] = {}
        d_mdl = np.zeros_like(data[bl])
        for band, band_slice in zip(bands, band_slices):
            d_mdl[tslice, band_slice], _, info_band = dspec.fourier_filter(
                data.times[tslice] * 24 * 60 * 60,
                data[bl][tslice, band_slice],
                wgts=wgts[bl][tslice, band_slice],
                filter_centers=[np.mean(fr_ranges[band]) / 1000],
                filter_half_widths=[np.diff(fr_ranges[band]) / 2 / 1000],
                mode='dpss_solve',
                eigenval_cutoff=[FR_EIGENVAL_CUTOFF], 
                suppression_factors=[FR_EIGENVAL_CUTOFF], 
                max_contiguous_edge_flags=len(data.times),
                cache=cache,
                filter_dims=0
            )
            info[bl][band] = info_band

        frf_data[bl] *= 0
        out = np.zeros_like(frf_data[bl])
        out[tslice, :] = np.where(wgts[bl][tslice, :] == 0, 0.0, d_mdl[tslice, :])
        frf_data[bl] = out

    return frf_data, info

# %%
def form_pseudostokes(data=None, flags=None, nsamples=None, pol_convention=hd.pol_convention, x_orientation=hd.telescope.get_x_orientation_from_feeds()):
    '''This function uses hera_pspec.pstokes to supplement existing datacontainers with pstokes I, Q, U, and/or V (where possible).'''
    for ap in data.antpairs():
        # loop over pseudo-stokes parameters
        iter_list = [('ee', 'nn', 'pI'), ('ee', 'nn', 'pQ'), ('en', 'ne', 'pU'), ('en', 'ne', 'pV')]
        for pol1, pol2, pstokes in iter_list:
            bl1 = ap + (pol1,)
            bl2 = ap + (pol2,)
            data_list = ([data[bl1], data[bl2]] if (data is not None) and (bl1 in data) and (bl2 in data) else None)
            flags_list = ([flags[bl1], flags[bl2]] if (flags is not None) and (bl1 in flags) and (bl2 in flags) else None)
            nsamples_list = ([nsamples[bl1], nsamples[bl2]] if (nsamples is not None) and (bl1 in nsamples) and (bl2 in nsamples) else None)

            # use hp.pstokes._combine_pol_arrays() to properly combine data/flags/nsamples in a pol_convention-aware way
            (combined_data, 
             combined_flags, 
             combined_nsamples) = hp.pstokes._combine_pol_arrays(pol1, pol2, pstokes, 
                                                                 pol_convention=pol_convention,
                                                                 data_list=data_list,
                                                                 flags_list=flags_list,
                                                                 nsamples_list=nsamples_list,
                                                                 x_orientation=x_orientation)
            # put results in original data containers
            if data is not None:
                data[ap + (pstokes,)] = combined_data
            if flags is not None:
                flags[ap + (pstokes,)] = combined_flags
            if nsamples is not None:
                nsamples[ap + (pstokes,)] = combined_nsamples

# %%
def timeavg_data(data, flags, nsamples, Navg=int(np.round(AVERAGING_TIME / (dt * NINTERLEAVE))), pols=None, tslice=slice(None), rephase=True):
    '''Performs coherent averaging of Navg integrations, rephasing to the common phase center.'''
    avg_data = datacontainer.DataContainer({})
    avg_flags = datacontainer.DataContainer({})
    avg_nsamples = datacontainer.DataContainer({})
    
    # perform time-averaging
    for bl in data:
        if (pols is not None) and (bl[2] not in pols):
            continue
        bl_vec =  data.antpos[bl[0]] - data.antpos[bl[1]]
        (avg_data[bl], 
         avg_flags[bl], 
         avg_nsamples[bl], 
         avg_lsts, 
         extra) = frf.timeavg_waterfall(data[bl][tslice, :], Navg,
                                        flags=np.zeros_like(flags[bl][tslice, :]),
                                        nsamples=nsamples[bl][tslice, :], 
                                        extra_arrays={'times': data.times[tslice]},
                                        lsts=data.lsts[tslice], freqs=data.freqs,
                                        rephase=rephase, bl_vec=bl_vec, verbose=False)
        avg_flags[bl][avg_nsamples[bl] == 0] = True # TODO: is this right???
    
    # attach relevant quantities to datacontainer
    for dc in (avg_data, avg_flags, avg_nsamples):
        dc.freqs = copy.deepcopy(data.freqs)
        dc.antpos = copy.deepcopy(data.antpos)
        dc.lsts = avg_lsts
        dc.times = extra['avg_times']
    return avg_data, avg_flags, avg_nsamples 

# %% [markdown]
# ### Figure out delay filter properties

# %%
bl_vec = (data.antpos[ANTPAIR[1]] - data.antpos[ANTPAIR[0]])
bl_len = np.linalg.norm(bl_vec[:2]) / constants.c

#Output the range of the delay filter, because we need it for graphs and estimating how much signal loss has occured in delay_filter_analysis.py
if PERFORM_DLY_FILT:
    with open(OUT_BL_DELAY_RANGE_FILE, "w") as file:
        file.write(f"BL_HW_(s), {max(DLY_FILT_HORIZON * bl_len * 1e9 + DLY_FILT_STANDOFF, DLY_FILT_MIN_DLY)}\n")
        #We multiply by 1e9, because bl_len is in seconds.

dly_filter_centers, dly_filter_half_widths = vis_clean.gen_filter_properties(
    ax='freq', horizon=DLY_FILT_HORIZON, standoff=DLY_FILT_STANDOFF, 
    min_dly=DLY_FILT_MIN_DLY, bl_len=bl_len
)
inpaint_filter_centers, inpaint_filter_half_widths = vis_clean.gen_filter_properties(
    ax='freq', horizon=INPAINT_HORIZON, standoff=INPAINT_STANDOFF, 
    min_dly=INPAINT_MIN_DLY, bl_len=bl_len
)

# %% [markdown]
# ### If desired, propagate flags on a channel to all times that will be coherently averaged together
# 
# This is primarily useful in the delay-filtered case with flags, where we want to avoid different interleaves having different flagging patterns.

# %%
def factored_flags(flag_array, tslice=slice(None)):
    '''If any channel is flagged in a boolean flag_array, flag all times for that channel unless the channel flag comes from an entirely flagged time.'''
    out_flags = np.zeros_like(flag_array)
    flagged_times = np.all(flag_array, axis=1)
    out_flags[flagged_times, :] = True
    if not np.all(flagged_times) :
        chan_flags = np.any(flag_array[~flagged_times], axis=0)
        out_flags[:, chan_flags] = True
    return out_flags

if FLAG_COHERENT_CHUNKS:
    Nchunk = int(AVERAGING_TIME // (dt * NINTERLEAVE)) * NINTERLEAVE
    for ci in range(int(np.ceil(flags[bl][tslice].shape[0] / Nchunk))):
        for bl in flags:
            flags[bl][tslice][ci * Nchunk:(ci + 1) * Nchunk] = factored_flags(flags[bl][tslice][ci * Nchunk:(ci + 1) * Nchunk], tslice=tslice)

# %% [markdown]
# ### Perform inpainting and/or delay-filtering

# %%
def per_band_avg_nsamples(nsamples, flags, band_slices):
    '''Create new datacontainer where nsamples has been averaged per-integration in band_slices.
    This is an approximate way to account for the fact that inpainted data is considered Nsamples=0.'''
    out_nsamples = copy.deepcopy(nsamples)
    for bl in out_nsamples:
        for band in band_slices:
            for i in range(out_nsamples[bl].shape[0]):
                out_nsamples[bl][i, band] = np.mean(out_nsamples[bl][i, band]) * (~np.any(flags[bl][i, band])).astype(float)
    return out_nsamples

# %%
def build_weights(data, flags, nsamples, wgt_by_avg_nsamples=False, band_slices=[]):
    '''Construct weights proportional to inverse noise variance (i.e. Nsamples / Autocorr^2).
    If wgt_by_avg_nsamples is True, average Nsamples in each subband (defined with band_slices).
    This avoids the introduction of spectral structure.'''
    wgts = copy.deepcopy(data)
    if wgt_by_avg_nsamples:
        nsamples_here = per_band_avg_nsamples(nsamples, flags, band_slices)
    else:
        nsamples_here = copy.deepcopy(nsamples)        
        
    for bl in wgts:               
        auto_bl = auto_antpair + bl[2:]
        wgts[bl] = np.where(flags[bl], 0, np.abs(data[auto_bl])**-2 * nsamples_here[bl])
        wgts[bl] /= np.abs(np.nanmean(np.where(flags[bl], np.nan, wgts[bl])))  # avoid dynamic range issues
        wgts[bl][~np.isfinite(wgts[bl])] = 0
    
    return wgts

# %%
# Build weights for delay-filter and/or inpainting that don't involve any Nsample averaging
freq_filt_wgts = build_weights(data, flags, nsamples) 

# %%
# Inpaint autocorrelations to allow for prediction of thermal noise on every channel
if not ALREADY_INPAINTED:
    _, data = delay_filter(data, freq_filt_wgts, inpaint_filter_centers, inpaint_filter_half_widths, INPAINT_EIGENVAL_CUTOFF, 
                           bls=[auto_antpair + (pol,) for pol in ['ee', 'nn']])

# %%
filt_data = copy.deepcopy(data)
filt_flags = copy.deepcopy(flags)
filt_nsamples = copy.deepcopy(nsamples)

# %%
# This cell replaces data with appropriate noise, which is useful for debugging
if USE_SIMULATED_NOISE:
    np.random.seed(21)

    reds = redcal.get_reds(filt_data.data_antpos, pols=filt_data.pols(), include_autos=True, bl_error_tol=2.0)
    red_inpainted = datacontainer.RedDataContainer(data, reds=reds)

    for bl in cross_bls:
        predicted_var = noise.predict_noise_variance_from_autos(bl, red_inpainted, nsamples=nsamples)
        predicted_var = np.where(~np.isfinite(predicted_var), 0, predicted_var)
        filt_data[bl] = np.sqrt(predicted_var) / 2**.5 * (np.random.randn(hd.Ntimes, hd.Nfreqs) + 1.0j * np.random.randn(hd.Ntimes, hd.Nfreqs))

# %%
# Inpaint crosses
if PERFORM_INPAINT:
    _, filt_data = delay_filter(data, freq_filt_wgts, inpaint_filter_centers, inpaint_filter_half_widths, INPAINT_EIGENVAL_CUTOFF)
    for bl in cross_bls:
        for band in [high_band, low_band]:
            if band.start >= band.stop:
                continue
            for i in range(filt_flags[bl].shape[0]):
                if not np.all(filt_flags[bl][i, band]):
                    filt_flags[bl][i, band] = False
inpainted = copy.deepcopy(filt_data)

# %%
# perform delay filtering on crosses
if PERFORM_DLY_FILT:
    filt_data, _ = delay_filter(inpainted, freq_filt_wgts, dly_filter_centers, dly_filter_half_widths, DLY_FILT_EIGENVAL_CUTOFF, zeros_where_zero_wgt=False)

# %%
# Recompute time filter flags, averaging nsamples in subbands if desired (this also applies to further processing)
if USE_BAND_AVG_NSAMPLES:
    filt_nsamples = per_band_avg_nsamples(filt_nsamples, filt_flags, band_slices)
    
time_filt_wgts = build_weights(filt_data, filt_flags, filt_nsamples)

# %%
# perform deinterleaving
deint_filt_data = filt_data.deinterleave(NINTERLEAVE, tslice=tslice)
deint_flags = filt_flags.deinterleave(NINTERLEAVE, tslice=tslice)
deint_nsamples = filt_nsamples.deinterleave(NINTERLEAVE, tslice=tslice)
deint_wgts = time_filt_wgts.deinterleave(NINTERLEAVE, tslice=tslice)

# %% [markdown]
# ### Figure out per-band fringe-rate filter ranges

# %%
# TODO: graduate this code into hera_cal

# load relevant FR spectrum vs. frequency and associated metadata
with h5py.File(FR_SPECTRA_FILE, "r") as h5f:
    metadata = h5f["metadata"]
    bl_to_index_map = {tuple(ap): int(index) for index, antpairs in metadata["baseline_groups"].items() for ap in antpairs}
    spectrum_freqs = metadata["frequencies_MHz"][()] * 1e6
    m_modes = metadata["erh_mode_integer_index"][()]
    if ANTPAIR in bl_to_index_map:
        mmode_spectrum = h5f["erh_mode_power_spectrum"][:, :, bl_to_index_map[ANTPAIR]]
    else:
        # If ANTPAIR is not in the FR_SPECTRA_FILE, but the reverse is, also reverse the spectrum
        mmode_spectrum = h5f["erh_mode_power_spectrum"][:, :, bl_to_index_map[ANTPAIR[::-1]]]
        m_modes *= -1

# %%
# Compute the band-averaged m-mode spectra.
interp_spec = interpolate.interp1d(spectrum_freqs, mmode_spectrum, kind="cubic", axis=1, fill_value="extrapoloate")(data.freqs)
band_avg_spec = {}
for band, bs in zip(bands, band_slices):
    taper = dspec.gen_window(TAPER, len(data.freqs[bs]))
    band_avg_spec[band] = np.average(interp_spec[:,bs], weights=taper**2, axis=1)

# %%
# Make one set of mixing matrices for the high resolution times for filter design.
full_times = data.times[tslice]
times_ks = (full_times - full_times[0] + np.median(np.diff(full_times))) * units.day.to(units.ks)
filt_frates = np.fft.fftshift(np.fft.fftfreq(times_ks.size, d=np.median(np.diff(times_ks))))
_m2f_mixer = frf.get_m2f_mixer(times_ks, m_modes)

# Use the deinterleaved time series for signal loss calculation.
deint_times = deint_filt_data[0].times
times_ks = (deint_times - deint_times[0] + np.median(np.diff(deint_times))) * units.day.to(units.ks)
frates = np.fft.fftshift(np.fft.fftfreq(times_ks.size, d=np.median(np.diff(times_ks))))
m2f_mixer = frf.get_m2f_mixer(times_ks, m_modes)

# %%
# TODO: graduate this code into hera_cal
# perform window-weighted average over each band, then get lower and upper quantiles
fr_ranges = {}
fr_profiles = {}
frf_losses = {}
xtalk_overlaps = {}
for band in bands:
    m_mode_spec = band_avg_spec[band]

    # Compute the filter bounds.
    band_avg_fr_spectrum = np.abs(np.einsum("fm,m,mf->f", _m2f_mixer, m_mode_spec, _m2f_mixer.T.conj()))
    band_avg_fr_spectrum /= np.sum(band_avg_fr_spectrum)
    cumsum_interpolator = interpolate.interp1d(np.cumsum(band_avg_fr_spectrum), filt_frates)
    fr_ranges[band] = cumsum_interpolator(FR_QUANTILE_LOW), cumsum_interpolator(FR_QUANTILE_HIGH)

    # Now compute the fringe-rate profile to be used for signal loss calculations.
    band_avg_fr_spectrum = np.abs(np.einsum("fm,m,mf->f", m2f_mixer, m_mode_spec, m2f_mixer.T.conj()))
    band_avg_fr_spectrum /= np.sum(band_avg_fr_spectrum)
    fr_profiles[band] = band_avg_fr_spectrum
    
    # account for overlap between FR=0 notch and main beam FRF
    def overlap_frs(frs1, frs2):
        start = np.maximum(frs1[0], frs2[0])
        end = np.minimum(frs1[1], frs2[1])
        return (start, end) if start < end else None
    frate_interpolator = interpolate.interp1d(frates, np.cumsum(band_avg_fr_spectrum), fill_value="extrapolate")
    frf_losses[band] = 1 - frate_interpolator(fr_ranges[band][1]) + frate_interpolator(fr_ranges[band][0])
    xtalk_overlaps[band] = overlap_frs(fr_ranges[band], [-XTALK_FR, XTALK_FR])
    if xtalk_overlaps[band] is not None:
        frf_losses[band] += frate_interpolator(xtalk_overlaps[band][1]) - frate_interpolator(xtalk_overlaps[band][0])

# %%
# Helper fft functions for signal loss calculation.
def FFT(data, axis):
    """Thin wrapper around fft stuff."""
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(data, axes=axis), axis=axis), axes=axis)

def IFFT(data, axis):
    """Thin wrapper around ifft stuff."""
    return np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(data, axes=axis), axis=axis), axes=axis)

# %%
# TODO: incorporate coherent average and time-interleaving to signal loss calculation.

# Update frf signal loss calculation.
frf_losses = {}
filter_design_matrices = {}
for band, band_slice in zip(bands, band_slices):
    # TODO: figure out how to handle frequency and polarization-dependent weights.
    # The weights change as a function of frequency (due to the inverse noise weighting),
    # so this isn't entirely accurate; will need to update in the future.
    # Also need to more carefully consider how to combine the weights across polarization,
    # and which precise set of weights to use when filtering the data.
    wgts = np.mean([deint_wgts[0][bl][:,band_slice] for bl in cross_bls if bl[2][0] == bl[2][1]], axis=(0,2))
    
    if np.all(wgts==0):
        frf_losses[band] = np.nan
        continue
    
    # Compute the filter center and filter half-width.
    fmin, fmax = fr_ranges[band]
    fc = 0.5 * (fmin+fmax)  # mHz
    fhw = 0.5 * np.abs(fmax-fmin)  # mHz
    
    # Generate the main-lobe and crosstalk time-time filter matrices.
    main_lobe_filt = frf.construct_filter(times_ks, fc, fhw, eigval_cutoff=FR_EIGENVAL_CUTOFF, wgts=wgts)
    xt_filt = np.eye(times_ks.size) - frf.construct_filter(times_ks, 0, XTALK_FR, eigval_cutoff=FR_EIGENVAL_CUTOFF, wgts=wgts)
    
    # Fourier transform to obtain the fringe-rate filter matrices.
    filter_xfer_matrix = FFT(IFFT(main_lobe_filt @ xt_filt, axis=0), axis=1)
    filter_design_matrices[band] = filter_xfer_matrix
    
    # Now compute the signal loss.
    filtered_power = np.sum(fr_profiles[band][None,:] * np.abs(filter_xfer_matrix)**2)
    unfiltered_power = np.sum(fr_profiles[band])
    frf_losses[band] = 1 - filtered_power / unfiltered_power

# %%
def show_FR_table():
    table = pd.DataFrame({'Band': np.arange(len(bands)) + 1,
                          'Frequency Range (MHz)': [f'{f0:.1f} — {f1:.1f}' for f0, f1 in zip(min_freqs, max_freqs)],
                          f'Main Beam {FR_QUANTILE_LOW:.0%} — {FR_QUANTILE_HIGH:.0%}<br>Kept Fringe Rates (mHz)': [f'{frs[0]:.3f} to {frs[1]:.3f}' for frs in fr_ranges.values()],
                          f'Signal Loss with<br>{-XTALK_FR} to {XTALK_FR} mHz X-Talk Filter': [f'{loss:.1%}' for loss in frf_losses.values()],
                         })
    return table.style.hide().to_html()

# %% [markdown]
# ### *Table 2: Fringe-Rate and Crosstalk Filtering Ranges and Losses*
# 
# The losses computed here are based on an extension of the framework from [Pascua+ 2025](https://iopscience.iop.org/article/10.3847/1538-4357/adc37d) to allow for non-uniform time weighting. These losses include contributions from both the main beam filter and the crosstalk filter, but do not include losses from coherent time averaging or the time-interleaved incoherent average.

# %%
HTML(show_FR_table())

# %%
# precomputation for rephasing before coherent averaging, ensuring all interleaves are rephased to the same lst
bl_vec = {bl: data.antpos[bl[0]] - data.antpos[bl[1]] for bl in [ap + (pol,) for ap in data.antpairs() 
                                                                 for pol in (utils._VISPOLS | set(utils.POL_STR2NUM_DICT))]}
for bl in list(bl_vec.keys()):
    if bl[0] != bl[1]:
        bl_vec[(bl[1], bl[0], bl[2])] = -bl_vec[bl]
Navg = int(AVERAGING_TIME // (dt * NINTERLEAVE))
n_avg_int = int(np.ceil(len(deint_filt_data[0].lsts) / Navg))
target_lsts = [np.mean(np.unwrap(np.sort(np.ravel([d.lsts[i * Navg:(i+1) * Navg] for d in deint_filt_data])))) for i in range(n_avg_int)]

# %% [markdown]
# ### Run time filtering, time averaging, and form pseudo-Stokes

# %%
# Initialize arrays for storing intermediate and final results for each interleave
deint_avg_data, deint_avg_flags, deint_avg_nsamples = [], [], []
deint_xtalk_filt_data, deint_frf_data = [], []

# List of dicts saying whether main lobe FRF was skipped or not
FRF_info = []
for d, f, n, w in zip(deint_filt_data, deint_flags, deint_nsamples, deint_wgts):
    # Perform full time-filtering
    if not SKIP_XTALK_AND_FRF:
        # Crosstalk filtering of FR=0 mode
        xtalk_filt_d = xtalk_filter(d, w, tslice=None)
        deint_xtalk_filt_data.append(xtalk_filt_d)
        
        # Main beam FRF and forming pseduostokes
        frf_d, info_stream = main_beam_FR_filter(xtalk_filt_d, w, tslice=None)
        FRF_info.append(info_stream)
        deint_frf_data.append(frf_d)
    else:
        deint_xtalk_filt_data.append(d)
        deint_frf_data.append(d)
    
    # Coherent time-averaging, rephasing to a set of lsts that's consistent across interleaves
    dlst = [target_lsts[i] - l for i in range(n_avg_int) for l in np.unwrap(d.lsts)[i * Navg:(i+1) * Navg]]
    rephased_frf_d = copy.deepcopy(deint_frf_data[-1])
    utils.lst_rephase(rephased_frf_d, bl_vec, d.freqs, dlst, lat=hd.telescope.location_lat_lon_alt_degrees[0], inplace=True)

    pstokes_pols = sorted([pol for pol in rephased_frf_d.pols() if utils.polstr2num(pol, x_orientation=hd.telescope.get_x_orientation_from_feeds()) > 0])
    avg_d, avg_f, avg_n = timeavg_data(rephased_frf_d, f, n, Navg=int(AVERAGING_TIME // (dt * NINTERLEAVE)), rephase=False)
    
    # Form pseudo-Stokes I, Q, U, and V from ee and nn
    form_pseudostokes(avg_d, avg_f, avg_n)

    # Store Results
    deint_avg_data.append(avg_d)
    deint_avg_flags.append(avg_f)
    deint_avg_nsamples.append(avg_n)    

# %% [markdown]
# ## Estimate total signal loss from FRF and coherent time average

# %%
# Retrieve the baseline vector and array latitude.
baseline = bl_vec[ANTPAIR + ("pI",)]
lat = hd.telescope.location.lat.rad

# Convert the pre-/post-average times to seconds
old_times = deint_filt_data[0].times
old_times = (old_times - old_times[0]) * units.day.to(units.s)
new_times = np.array(deint_avg_data[0].times)
new_times = (new_times - deint_filt_data[0].times[0]) * units.day.to(units.s)
new_inttime = AVERAGING_TIME

# %%
frf_plus_tavg_losses = {}

for band, band_slice in zip(bands, band_slices):
    wgts = np.mean([deint_wgts[0][bl][:,band_slice] for bl in cross_bls if bl[2][0] == bl[2][1]], axis=(0,2))
    
    if np.all(wgts==0):
        frf_plus_tavg_losses[band] = np.nan
        continue

    # Retrieve the FRF filter matrix.
    filter_xfer_matrix = filter_design_matrices[band]

    # Compute the band-averaged FR profile.
    freqs = data.freqs[band_slice]
    avg_spec = band_avg_spec[band]
    fr_profile = np.diag(m2f_mixer @ (avg_spec[:,None] * m2f_mixer.T.conj()))
    
    # Compute the filtered fringe-rate profile and convert to a time-time covariance.
    filt_cov = FFT(IFFT(filter_xfer_matrix @ (fr_profile[:,None] * filter_xfer_matrix.T.conj()), axis=1), axis=0)

    # Retrieve n_samples and flags for this band.
    n_samples = deint_nsamples[0][ANTPAIR]['nn'][:,band_slice] + deint_nsamples[0][ANTPAIR]['ee'][:,band_slice]
    _flags = deint_flags[0][ANTPAIR]['ee'][:,band_slice] | deint_flags[0][ANTPAIR]['nn'][:,band_slice]

    # Take the band-averaged n_samples and flags to avoid NaNs from fully-flagged channels.
    n_samples = n_samples.mean(axis=1)
    _flags = _flags.mean(axis=1).astype(bool)
    
    # Compute the time-average design matrix at the center of the band.
    design_mat = frf.get_coherent_avg_design_matrix(
        baseline, lat, freqs.mean(), old_times, new_times, n_samples, new_inttime, _flags
    )[0]

    # Now compute the signal loss.
    filtered_power = np.abs(filt_cov * (design_mat.T @ design_mat.conj())).sum() / design_mat.shape[0]
    unfiltered_power = np.abs(fr_profile).mean()
    frf_plus_tavg_losses[band] = 1 - filtered_power / unfiltered_power

# %%
def show_loss_table():
    table = pd.DataFrame({'Band': np.arange(len(bands)) + 1,
                          'Frequency Range (MHz)': [f'{f0:.1f} — {f1:.1f}' for f0, f1 in zip(min_freqs, max_freqs)],
                          f'Main Beam {FR_QUANTILE_LOW:.0%} — {FR_QUANTILE_HIGH:.0%}<br>Kept Fringe Rates (mHz)': [f'{frs[0]:.3f} to {frs[1]:.3f}' for frs in fr_ranges.values()],
                          f'Signal Loss with<br>{-XTALK_FR} to {XTALK_FR} mHz X-Talk Filter': [f'{loss:.1%}' for loss in frf_losses.values()],
                          f'Signal Loss with<br>{int(AVERAGING_TIME)} s Coherent Average': [f'{loss:.1%}' for loss in frf_plus_tavg_losses.values()],
                         })
    return table.style.hide().to_html()

# %%
HTML(show_loss_table())

# %% [markdown]
# # Table 3: Signal Loss Estimates from Fringe-Rate Filters and Coherent Time Average
# 
# Similar to Table 2, these losses are computed based on an extension to the framework from [Pascua+ 2025](https://iopscience.iop.org/article/10.3847/1538-4357/adc37d), using weights that are time-variable (but uniform in frequency). Note that the rightmost column is the **total** loss from the main lobe filter, the crosstalk filter, and the coherent time average.

# %% [markdown]
# ### Compute correction factor for coherent averaging, either approximately with mode counting or more exactly with the FRF noise covariances (set USE_CORR_MATRIX=True)

# %%
# TODO: this function should probably be graduated into hera_pspec or hera_cal

def dpss_coherent_avg_correction(spw):
    '''This function computes an approximate correction to the noise calculation after fringe-rate filtering. It assumes the that number of integrations
    that are coherently averaged together is equal the ratio of the number of integrations per interleave divided by the number of FR modes kept. This
    is then used to correct the calculation done in hera_pspec, which doesn't know about the FRF. The actual noise power spectrum is reduced by this factor
    compared to what naively comes out of hera_pspec. However, when performing incoherent averaging of power spectra, one needs to raise noise power spectrum
    by the square-root of this factor to account for the correlations between coherently-averaged power spectrum bins.'''
    if SKIP_XTALK_AND_FRF:
        coherent_avg_correction_factor = 1.0
    else: 
        band = bands[spw]
        time_in_seconds = (deint_filt_data[0].times - deint_filt_data[0].times.min()) * 60 * 60 * 24  # time array in seconds
        time_filters = dspec.dpss_operator(time_in_seconds, [np.mean(fr_ranges[band]) / 1000], 
                                           [np.diff(fr_ranges[band]) / 2 / 1000], eigenval_cutoff=[FR_EIGENVAL_CUTOFF])[0].real

        # count the effective number of integrations that go into each coherent average, accounting for overlap with the xtalk filter
        if xtalk_overlaps[band] is None:
            actual_integrations_per_coherent_avg = time_filters.shape[0] / time_filters.shape[1]  # ratio of total number of DPSS FR modes to modes kept after filtering
        else:
            overlap_filters = dspec.dpss_operator(time_in_seconds, [np.mean(xtalk_overlaps[band]) / 1000], 
                                                  [np.diff(xtalk_overlaps[band]) / 2 / 1000], eigenval_cutoff=[FR_EIGENVAL_CUTOFF])[0].real
            actual_integrations_per_coherent_avg = time_filters.shape[0] / (time_filters.shape[1]  - overlap_filters.shape[1])
        
        integrations_per_coherent_avg = int(AVERAGING_TIME // (dt * NINTERLEAVE))
        coherent_avg_correction_factor = actual_integrations_per_coherent_avg / integrations_per_coherent_avg
    return coherent_avg_correction_factor

# %%
def get_frop_wrapper(mode="main_lobe", pol="nn", stream_ind=0, band_ind=0, t_avg=AVERAGING_TIME, 
                     rephase=True, wgt_tavg_by_nsample=True, bl_vec=None,
                     dlst=None, coherent_avg=True, times=None,
                     freq_decimation=CORR_MATRIX_FREQ_DECIMATION):
    """
    This wraps hera_cal.frf.get_frop_for_noise using specific information from this notebook so that we can do
    post FRF time-time visibility covariance calculation. It returns an operator that can be used on a dynamic 
    spectrum to filter it down the time axis.
    """
    bl=(ANTPAIR[0], ANTPAIR[1], pol)
    
    band = bands[band_ind]
    band_slice = band_slices[band_ind]
    if freq_decimation > 1:
        band_slice = slice(
            band_slice.start, 
            band_slice.stop, 
            freq_decimation
        )

    weights=deint_wgts[stream_ind][bl][:, band_slice]
    nsamples = deint_nsamples[stream_ind][bl][:, band_slice]
    
    if times is None:
        times = np.modf(single_bl_times[tslice][stream_ind::NINTERLEAVE])[0] * 24 * 3600
        times = times - times[0]

    if mode == "main_lobe":
        filter_cent_use = [np.mean(fr_ranges[band]) / 1000]
        filter_half_wid_use = [np.diff(fr_ranges[band]) / 2 / 1000]
    elif mode == "xtalk":
        filter_cent_use = [0]
        filter_half_wid_use=[XTALK_FR / 1000]
    
    frop = frf.get_frop_for_noise(times, filter_cent_use=filter_cent_use, 
                                  filter_half_wid_use=filter_half_wid_use, 
                                  freqs=data.freqs[band_slice], t_avg=t_avg,
                                  eigenval_cutoff=FR_EIGENVAL_CUTOFF, weights=weights,
                                  rephase=rephase, wgt_tavg_by_nsample=wgt_tavg_by_nsample,
                                  nsamples=nsamples, bl_vec=bl_vec, dlst=dlst,
                                  coherent_avg=coherent_avg)
    
    if mode == "xtalk":
        frop = np.eye(len(times))[:, :, None] - frop
    return frop

# %%
coherent_avg_correction_factors = []
for spw, band in enumerate(bands):
    freq_slice = band_slices[spw]
    decimated_freq_slice = slice(
        freq_slice.start, 
        freq_slice.stop, 
        CORR_MATRIX_FREQ_DECIMATION
    )
        
    if USE_CORR_MATRIX and not SKIP_XTALK_AND_FRF:
        corr_factors = np.zeros(NINTERLEAVE, dtype=float)
        known_bad_streams = np.zeros(NINTERLEAVE, dtype=bool)
        for stream_ind in range(NINTERLEAVE):
            times=deint_filt_data[stream_ind].times * 24 * 3600    
            cov_here = 0
            num_skipped_pols = 0
            for pol in ("ee", "nn"):
                cross_antpairpol = ANTPAIR + (pol,)
                info_here = FRF_info[stream_ind][cross_antpairpol][bands[spw]]["status"]["axis_0"]
                info_array = np.array([info_here[key] for key in info_here])
                
                # Assuming any skips mean the whole deal is off. 
                # Since we're decimating usually by a factor of 10, this should not often be affected by the odd fully flagged channel here or there.
                Nfreqs_spw = freq_slice.stop - freq_slice.start
                info_slice = slice(0, 
                                   Nfreqs_spw,
                                   CORR_MATRIX_FREQ_DECIMATION)
                if np.any(info_array[info_slice] == "skipped"): 
                    num_skipped_pols += 1
                    num_skip_freq = np.count_nonzero(info_array == "skipped")
                    frac_skip_freq = num_skip_freq / (freq_slice.stop - freq_slice.start)
                    percent_skip_freq = frac_skip_freq * 100
                    percent_skip_freq = "%.2f" % percent_skip_freq
                    print(f"{percent_skip_freq}\% of frequencies were skipped in band {spw + 1}, stream {stream_ind}, pol {pol}, skipping covariance calculation")
                    continue
                else:

                    var = frf.prep_var_for_frop(deint_filt_data[stream_ind],
                                                deint_nsamples[stream_ind],
                                                deint_wgts[stream_ind],
                                                cross_antpairpol,
                                                decimated_freq_slice,
                                                auto_ant=0)

                    main_lobe_frop = get_frop_wrapper(pol=pol, stream_ind=stream_ind, band_ind=spw,
                                                      dlst=dlst, bl_vec=bl_vec[cross_antpairpol],
                                                      times=times)

                        

                    # Calculate whether the EW-baseline projection is short enough to care about the notch
                    EW_proj = data.antpos[ANTPAIR[0]][0] - data.antpos[ANTPAIR[1]][0]
                    if np.abs(EW_proj) < CORR_MATRIX_NOTCH_CUTOFF:
                        xtalk_frop = get_frop_wrapper(pol=pol, stream_ind=stream_ind, band_ind=spw,
                                                      dlst=dlst, bl_vec=bl_vec[cross_antpairpol],
                                                      times=times, rephase=False, mode="xtalk", coherent_avg=True,
                                                      t_avg=times[1] - times[0])


                        frop = np.zeros_like(main_lobe_frop)
                        Nfreqs_band = freq_slice.stop - freq_slice.start
                        Nfreqs_calc = int(np.ceil(Nfreqs_band / CORR_MATRIX_FREQ_DECIMATION))

                        for freq_ind in range(Nfreqs_calc):
                            frop[:, :, freq_ind] = np.tensordot(main_lobe_frop[:, :, freq_ind],
                                                                xtalk_frop[:, :, freq_ind],
                                                                axes=1)
                    else:
                        frop = main_lobe_frop
                cov_here += frf.get_FRF_cov(frop, var)  # computes covariance for pI by adding ee and nn
            if hd.pol_convention == "avg" and num_skipped_pols == 0:
                cov_here /= 4

            # Don't use times that are all flagged
            ALLed_flags = np.all(deint_avg_flags[stream_ind][ANTPAIR + ('pI',)][:, freq_slice], axis=1)

            if num_skipped_pols < 2 and not np.all(ALLed_flags): # Don't allow an empty slice
                corr_factors[stream_ind] = frf.get_correction_factor_from_cov(cov_here, tslc=(~ALLed_flags))
            else:
                print(f"No valid pI data in band {spw + 1}, stream {stream_ind}. Setting this correction factor to nan.")
                corr_factors[stream_ind] = np.nan
                known_bad_streams[stream_ind] = True
                
        if np.all(known_bad_streams): # We think we understand why they're all bad, set to nan
            print(f"No valid correction factors found for band {spw + 1}, setting to nan.")
            corr_factor = np.nan
        else:
            which_finite = np.isfinite(corr_factors[~known_bad_streams])
            # Catches potential baddies from time slicing corner cases that we haven't thought of, should error
            assert np.all(which_finite), f'For band {band}, corr_factors={corr_factors}, known_bad_streams={known_bad_streams}'
            corr_factor = np.mean(corr_factors[~known_bad_streams]) 
        coherent_avg_correction_factors.append(corr_factor)

    else:
        # use the mode-counting method, which is simpler but less accurate
        coherent_avg_correction_factors.append(dpss_coherent_avg_correction(spw))

# %% [markdown]
# ### *Figure 2: Waterfalls Before Delay Filtering and/or Inpainting*

# %%
if PLOT:
    plot_waterfall(data, flags=flags)
    plot_real_delay_vs_lst(data, flags=flags, xlim=[-3999, 3999], clim=[-1e4, 1e4])
    plot_dly_vs_fr(data, xlim=[-1999, 1999], clim=[1e0, 1e5])

# %% [markdown]
# ### *Figure 3: Waterfalls After Delay Filtering and/or Inpainting*

# %%
if PLOT and (PERFORM_DLY_FILT or PERFORM_INPAINT):
    plot_waterfall(filt_data, flags=filt_flags, nsamples=nsamples)
    plot_real_delay_vs_lst(filt_data, flags=filt_flags, xlim=[-3999, 3999], clim=[-1e4, 1e4])
    plot_dly_vs_fr(filt_data, xlim=[-1999, 1999], clim=[1e0, 1e5])

# %% [markdown]
# ### *Figure 4: First Set of De-Interleaved Waterfalls after Cross-Talk Filtering*

# %%
if PLOT and not SKIP_XTALK_AND_FRF:
    plot_waterfall(deint_xtalk_filt_data[0], flags=deint_flags[0], nsamples=deint_nsamples[0], tslice=None)
    plot_real_delay_vs_lst(deint_xtalk_filt_data[0], flags=deint_flags[0], xlim=[-3999, 3999], clim=[-1e4, 1e4], tslice=None)
    plot_dly_vs_fr(deint_xtalk_filt_data[0], xlim=[-1999, 1999], clim=[1e0, 1e5], tslice=None)

# %% [markdown]
# ### *Figure 5: First Set of De-Interleaved Waterfalls after Main-Beam Fringe-Rate Filtering*

# %%
if PLOT and not SKIP_XTALK_AND_FRF:
    plot_waterfall(deint_frf_data[0], flags=deint_flags[0], nsamples=deint_nsamples[0], tslice=None)
    plot_real_delay_vs_lst(deint_frf_data[0], flags=deint_flags[0], xlim=[-3999, 3999], clim=[-2e2, 2e2], linthresh=1, tslice=None)
    plot_dly_vs_fr(deint_frf_data[0], xlim=[-1999, 1999], clim=[1e0, 1e5], tslice=None)

# %% [markdown]
# ### *Figure 6: First Set of De-Interleaved Waterfalls after Coherent Time Averaging*

# %%
if PLOT:
    plot_waterfall(deint_avg_data[0], bl=(ANTPAIR + ('ee',)), flags=deint_avg_flags[0], nsamples=deint_avg_nsamples[0], tslice=None)
    plot_real_delay_vs_lst(deint_avg_data[0], bl=(ANTPAIR + ('ee',)), flags=deint_avg_flags[0], xlim=[-3999, 3999], clim=[-2e2, 2e2], linthresh=1, tslice=None)
    plot_dly_vs_fr(deint_avg_data[0], bl=(ANTPAIR + ('ee',)), xlim=[-1999, 1999], clim=[1e0, 1e5], tslice=None)

# %% [markdown]
# ### *Figure 7: First Set of De-Interleaved Waterfalls after Forming Pseudo-Stokes I*

# %%
if PLOT:
    plot_waterfall(deint_avg_data[0], bl=(ANTPAIR + ('pI',)), flags=deint_avg_flags[0], nsamples=deint_avg_nsamples[0], tslice=None)
    plot_real_delay_vs_lst(deint_avg_data[0], bl=(ANTPAIR + ('pI',)), flags=deint_avg_flags[0], xlim=[-3999, 3999], clim=[-2e2, 2e2], linthresh=1, tslice=None)
    plot_dly_vs_fr(deint_avg_data[0], bl=(ANTPAIR + ('pI',)), xlim=[-1999, 1999], clim=[1e0, 1e5], tslice=None)

# %% [markdown]
# ## Power Spectrum Estimation

# %% [markdown]
# ### Prepare for power spectrum estimation

# %%
# put results back into HERAData objects for use in hera_pspec (which works with UVData objects)
hds = []
for avg_data, avg_flags, avg_nsamples in zip(deint_avg_data, deint_avg_flags, deint_avg_nsamples):
    avg_hd = copy.deepcopy(hd)
    
    # select the right number of times and update time and lst arrays
    avg_hd.select(times=np.unique(avg_hd.time_array)[:len(avg_data.times)])
    for ap in avg_hd.get_antpairs():
        blt_slice = avg_hd._blt_slices[ap]
        avg_hd.time_array[blt_slice] = avg_data.times
        avg_hd.lst_array[blt_slice] = avg_data.lsts

    # update polarizations
    pstokes_pols = sorted([pol for pol in avg_data.pols() if utils.polstr2num(pol, x_orientation=hd.telescope.get_x_orientation_from_feeds()) > 0])
    avg_hd.polarization_array = np.array([utils.polstr2num(pol) for pol in pstokes_pols])
    avg_hd._determine_pol_indexing()
    
    # update pstokes only data in avg_hd
    avg_data_for_update = datacontainer.DataContainer({bl: avg_data[bl] for bl in avg_data if bl[2] in pstokes_pols})
    avg_flags_for_update = datacontainer.DataContainer({bl: avg_flags[bl] for bl in avg_flags if bl[2] in pstokes_pols})
    avg_nsamples_for_update = datacontainer.DataContainer({bl: avg_nsamples[bl] for bl in avg_nsamples if bl[2] in pstokes_pols})
    avg_hd.update(data=avg_data_for_update, flags=avg_flags_for_update, nsamples=avg_nsamples_for_update)
    
    # add in autocorrelations copies for all antennas for use in noise calculations
    auto_aps = [ap for ap in avg_hd.get_antpairs() if ap[0] == ap[1]]
    for ant in ANTPAIR:
        if (ant, ant) not in auto_aps:
            avg_hd_copy = copy.deepcopy(avg_hd)
            avg_hd_copy.select(bls=[auto_aps][0])
            avg_hd_copy.ant_1_array = np.full_like(avg_hd_copy.ant_1_array, ant)
            avg_hd_copy.ant_2_array = np.full_like(avg_hd_copy.ant_2_array, ant)
            avg_hd_copy.baseline_array = uvutils.antnums_to_baseline(avg_hd_copy.ant_1_array, avg_hd_copy.ant_2_array, Nants_telescope=avg_hd_copy.Nants_telescope)
            avg_hd.fast_concat(avg_hd_copy, 'blt', inplace=True)
    
    hds.append(avg_hd)

# %%
# Load uvbeam file
uvb = UVBeam()
uvb.read(EFIELD_HEALPIX_BEAM_FILE)

# convert to pstokes and peak-normalized power beam
uvb_ps = uvb.efield_to_pstokes(inplace=False)
uvb_ps.peak_normalize()
uvb.efield_to_power()
uvb.peak_normalize()

# %% [markdown]
# ### Estimate power spectra for all unique interleaved pairs

# %%
# Estimate power spectrum
cosmo = hp.conversions.Cosmo_Conversions()
pspecbeam = hp.pspecbeam.PSpecBeamUV(uvb_ps, cosmo=cosmo)

# %%
band = bands[0]
time_in_seconds = (deint_filt_data[0].times - deint_filt_data[0].times.min()) * 60 * 60 * 24  # time array in seconds
time_filters = dspec.dpss_operator(time_in_seconds, [np.mean(fr_ranges[band]) / 1000], [np.diff(fr_ranges[band]) / 2 / 1000], eigenval_cutoff=[FR_EIGENVAL_CUTOFF])[0].real

# %%
# Compute power spectra for all unique pairs of interleaves
uvps = []
for ind1, hd1 in enumerate(hds):
    for ind2, hd2 in enumerate((hds[ind1:] if INCLUDE_INTERLEAVE_AUTO_PS else hds[ind1 + 1:])):
        # Compute power spectrum
        ds = hp.PSpecData(dsets=[copy.deepcopy(hd1), copy.deepcopy(hd2)], beam=pspecbeam)
        ds.Jy_to_mK()
        uvp = ds.pspec([ANTPAIR], [ANTPAIR], dsets=(0, 1), 
                       pols=[utils.polnum2str(pol) for pol in hd1.polarization_array], 
                       spw_ranges=[(bs.start, bs.stop) for bs in band_slices],
                       taper=TAPER, store_window=STORE_WINDOW_FUNCTIONS, verbose=False)
        
        # Figure out error bars using autocorrelations
        auto_Tsys = hp.utils.uvd_to_Tsys((hd1 + hd2), pspecbeam)
        hp.utils.uvp_noise_error(uvp, auto_Tsys, err_type=['P_N'])
        
        # append to list of power spectra
        uvps.append(uvp)

# %% [markdown]
# ### A note on estimating $P_{SN}$:
# 
# [Tan et al. (2021)](https://arxiv.org/abs/2103.09941) showed that $P_{SN}^2 = \sqrt{2}P_S P_N + P_N^2$ (Eq. 30). The challenge is estimating $P_S$. Any power spectrum measurement has both noise and signal in it, so even with the imposition of the prior that $P_S$ is real and non-negative, we expect double counting of the noise. [Tan et al. (2021)](https://arxiv.org/abs/2103.09941) shows that the expectation value of the 1st term on the LHS of Eq. 30 has an expectation value of $P_N^2 / \sqrt{\pi}$ for Gaussian-distributed power spectra (see Eq. 31) and $P_N^2 / 2$ for Laplacian-distributed power spectra. The former is appropriate after sufficient time averaging, the latter for single integrations.
# 
# In this code, we estimate $P_S$ from all *other* interleaves, which reduces the overcount of $P_N$ in the formula for $P_{SN}$ by a factor of `(len(uvps) - 1)`. It also makes the histogram of $P(k) / P_{SN}$ closer to the theoretical expectation, since we never divide $P(k)$ by itself.

# %%
# TODO: graduate this code into hera_pspec

# Apply coherent_avg_correction_factor and compute P_SN
for spw, band in enumerate(bands):
    for i, uvp in enumerate(uvps):
        # loop over all pols, but only this spw
        for key in uvp.get_all_keys():
            if key[0] != spw:
                continue
        
            # apply coherent average correction due to integrations not being independent after FRF
            P_N = uvp.get_stats('P_N', key) / coherent_avg_correction_factors[spw]
            P_N[~np.isfinite(P_N)] = np.inf
            uvp.set_stats('P_N', key, P_N)
            
            # use other interleaves to estimate P_S
            P_S = np.mean([uvp2.get_data(key).real for j, uvp2 in enumerate(uvps) if i != j], axis=0)
            P_SN = np.sqrt((np.sqrt(2) * np.where(P_S > 0, P_S, 0) * P_N + P_N**2))  # Tan et al. 2021, Eq. 30
            # Apply P_SN Correction for Laplacian statistics (see note above)
            P_SN = (P_SN**2 - .5 / (len(uvps) - 1) * P_N**2)**.5 
            P_SN[~np.isfinite(P_SN)] = np.inf
            uvp.set_stats('P_SN', key, P_SN)
# %%
# TODO: graduate this code into hera_pspec

# Optional union-of-ranges override for the final time-average LST selection (BAND_STR syntax, hours).
# When None/empty, falls back to f"{LST_MIN}~{LST_MAX}".
PSPEC_LST_AVG_RANGES = globals().get('PSPEC_LST_AVG_RANGES', f"{LST_MIN}~{LST_MAX}")
lst_ranges_rad = parse_lst_ranges_rad(PSPEC_LST_AVG_RANGES)

uvps_time_avg = []
for uvp in uvps:
    mask = lst_in_ranges(uvp.lst_avg_array, lst_ranges_rad)
    lst_subset = uvp.lst_avg_array[mask]
    uvp_tavg = uvp.select(lsts=lst_subset, polpairs=[('pI', 'pI')], inplace=False)
    uvp_tavg.average_spectra(time_avg=True, error_weights='P_N', error_field=['P_N', 'P_SN'], inplace=True)
    uvps_time_avg.append(uvp_tavg)

for spw, band in enumerate(bands):
    coherent_avg_correction_factor = coherent_avg_correction_factors[spw]
    for i, uvp in enumerate(uvps_time_avg):   
        # loop over all pols, but only this spw
        for key in uvp.get_all_keys():
            if key[0] != spw:
                continue
            
            # Update P_N and P_SN for incoherent time average
            P_N = uvp.get_stats('P_N', key) * (coherent_avg_correction_factor)**.5
            P_N[~np.isfinite(P_N)] = np.inf    
            uvp.set_stats('P_N', key, P_N)
            P_S = np.mean([uvp2.get_data(key).real for j, uvp2 in enumerate(uvps_time_avg) if i != j], axis=0)      
            P_SN = np.sqrt((np.sqrt(2) * np.where(P_S > 0, P_S, 0) * P_N + P_N**2))  # Tan et al. 2021, Eq. 30     
            # Apply P_SN Correction for Gaussian statistics (see note above)
            P_SN = (P_SN**2 - 1./ np.sqrt(np.pi) / (len(uvps_time_avg) - 1) * P_N**2)**.5 
            P_SN[~np.isfinite(P_SN)] = np.inf
            uvp.set_stats('P_SN', key, P_SN)

# %% [markdown]
# ### Average Over Interleaves

# %%
# combine all interleaves into UVPSpec object
interleaved_uvp = hp.uvpspec.recursive_combine_uvpspec(uvps)

# select each individual time and average interleaves together incoherently
to_recombine = []
for times in interleaved_uvp.time_avg_array.reshape(-1, len(uvps)):
    to_recombine.append(interleaved_uvp.select(times=times, inplace=False))
    to_recombine[-1].average_spectra(time_avg=True, error_weights='P_N', error_field=['P_N', 'P_SN'])

# combine all single-integration UVPSpec objects
interleaved_uvp = hp.uvpspec.recursive_combine_uvpspec(to_recombine)

# %%
# Perform incoherent time-averaging across interleaves
uvp_interleave = reduce(lambda x, y: x + y, uvps_time_avg)
uvp_avg_all = uvp_interleave.average_spectra(time_avg=True, error_weights='P_N', error_field=['P_N', 'P_SN'], inplace=False)

# %%
# Artificially set the time_2_array to time_1_array because all interleaves are at the "same" time
uvp_interleave.time_2_array = uvp_interleave.time_1_array
uvp_avg_all.time_2_array = uvp_avg_all.time_1_array

# %% [markdown]
# ### Correct for FRF Signal Loss

# %%
hp.loss.apply_bias_correction(
    interleaved_uvp, data_bias={spw: (1.0 - loss)**-1 if np.isfinite(loss) else 1.0 for spw, loss in enumerate(frf_losses.values())}
)
hp.loss.apply_bias_correction(
    uvp_avg_all, data_bias={spw: (1.0 - loss)**-1 if np.isfinite(loss) else 1.0 for spw, loss in enumerate(frf_losses.values())}
)

# %% [markdown]
# ### Power Spectrum Plotting Code

# %%
# average all interleaved LSTs together for plotting
all_lsts = []
for uvp in uvps:
    if np.mean(np.unwrap(uvp.lst_avg_array)) - np.mean(np.unwrap(uvps[0].lst_avg_array)) > np.pi:
        all_lsts.append(np.unwrap(uvp.lst_avg_array) - 2 * np.pi)
    elif np.mean(np.unwrap(uvp.lst_avg_array)) - np.mean(np.unwrap(uvps[0].lst_avg_array)) < -np.pi:        
        all_lsts.append(np.unwrap(uvp.lst_avg_array) + 2 * np.pi)
    else:
        all_lsts.append(np.unwrap(uvp.lst_avg_array))
avg_pspec_lsts = (np.mean(all_lsts, axis=0) % (2 * np.pi))

# %%
# Compute SNRs for plotting
SNRs = []
tavg_SNRs = []
for spw, band in enumerate(bands):
    key = (spw, (ANTPAIR, ANTPAIR), ('pI', 'pI')) 
    for i, uvp in enumerate(uvps):
        high_dlys = np.abs(uvp.get_dlys(key[0]) * 1e9) > 1000
        SNRs.append(np.ravel((uvp.get_data(key) / uvp.get_stats('P_N', key).real)[:, high_dlys]))
    for i, uvp in enumerate(uvps_time_avg):
        high_dlys = np.abs(uvp.get_dlys(key[0]) * 1e9) > 1000
        tavg_SNRs.append(np.ravel((uvp.get_data(key) / uvp.get_stats('P_N', key).real)[:, high_dlys]))

# %%
def plot_Pk_vs_LST(clim=None, xlim=[-2999, 2999], pol='pI'):
    '''Plots the real part of the power spectrum from each band as a function of LST and delay for each band.'''
    
    lsts = np.where(avg_pspec_lsts > avg_pspec_lsts[-1], avg_pspec_lsts - 2 * np.pi, avg_pspec_lsts) * 12 / np.pi
    fig, axes = plt.subplots(1, len(bands), figsize=(28, 12), sharex=True, sharey=True, gridspec_kw={'wspace': .03}, dpi=100)
    for spw, (ax, band, band_slice) in enumerate(zip(axes, bands, band_slices)):

        key = (spw, (ANTPAIR, ANTPAIR), (pol, pol))  
        pk_avg = interleaved_uvp.get_data(key).real
#        np.mean([uvp.get_data(key) for uvp in uvps], axis=0).real
        delays = interleaved_uvp.get_dlys(key[0]) * 1e9
#        uvps[0].get_dlys(key[0]) * 1e9

        if spw == 0:
            _to_plot = copy.deepcopy(np.where(np.isfinite(pk_avg), pk_avg, np.nan))
            _to_plot = np.where(_to_plot == 0, np.nan, _to_plot)
        im = ax.imshow(np.abs(np.where(np.isfinite(pk_avg), pk_avg, np.nan)), 
                       interpolation='none', aspect='auto', cmap='inferno', 
                       norm=matplotlib.colors.LogNorm(vmin=(clim[0] if clim is not None else np.nanmin(np.abs(_to_plot))), 
                                                      vmax=(clim[1] if clim is not None else np.nanmax(np.abs(_to_plot)))),
                       extent=[delays[0], delays[-1], lsts[-1], lsts[0]])

        for multiple in [1, -1]:
            ax.axvline(multiple * dly_filter_half_widths[0] * 1e9, ls='--', color='k')
            ax.axvline(multiple * inpaint_filter_half_widths[0] * 1e9, ls=':', color='k')
        ax.set_xlim(xlim)
        ax.set_title(f'Band {spw+1}:\n{band[0]}—{band[1]} MHz', fontsize=10)
        ax.set_xlabel('Delay (ns)')
        if spw == 0:
            ax.set_ylabel('LST (Hours)')
            ax.set_yticklabels([f'{(int(val) if np.isclose(val, int(val)) else val) % 24}' for val in ax.get_yticks()])

    plt.colorbar(im, ax=axes, pad=.02, aspect=40, extend='both', label=f'{pol} ' + r'|Re[$P(k)$]| (mK$^2$ $h^{-3}$ Mpc$^3$)')

# %%
def plot_Pk_SNR_vs_LST(clim=[-5, 5], xlim = [-2999, 2999], func=np.real):
    '''Plots the real power spectrum SNR (normalized by P_N, not P_SN) as a function of LST and delay for each band.'''
    
    lsts = np.where(data.lsts[tslice] > data.lsts[tslice][-1], data.lsts[tslice] - 2 * np.pi, data.lsts[tslice]) * 12 / np.pi
    fig, axes = plt.subplots(1, len(bands), figsize=(28, 12), sharex=True, sharey=True, gridspec_kw={'wspace': .03}, dpi=100)
    for spw, (ax, band, band_slice) in enumerate(zip(axes, bands, band_slices)):

        key = (spw, (ANTPAIR, ANTPAIR), ('pI', 'pI'))  
        pk_avg = func(interleaved_uvp.get_data(key))
#        func(np.mean([uvp.get_data(key) for uvp in uvps], axis=0))
        delays = interleaved_uvp.get_dlys(key[0]) * 1e9
        P_N = np.abs(interleaved_uvp.get_stats("P_N", key)) #np.mean([np.abs(uvp.get_stats('P_N', key)) for uvp in uvps], axis=0)

        SNR = pk_avg / P_N# / np.sqrt(len(uvps)))

        im = ax.imshow(SNR, interpolation='none', aspect='auto', cmap='bwr', 
                       vmin=clim[0], vmax=clim[1],
                       extent=[delays[0], delays[-1], lsts[-1], lsts[0]])
        for multiple in [1, -1]:
            ax.axvline(multiple * dly_filter_half_widths[0] * 1e9, ls='--', color='k')
            ax.axvline(multiple * inpaint_filter_half_widths[0] * 1e9, ls=':', color='k')
        ax.set_xlim(xlim)
        ax.set_title(f'Band {spw+1}:\n{band[0]}—{band[1]} MHz', fontsize=10)
        ax.set_xlabel('Delay (ns)')
        if spw == 0:
            ax.set_ylabel('LST (Hours)')
            ax.set_yticklabels([f'{(int(val) if np.isclose(val, int(val)) else val) % 24}' for val in ax.get_yticks()])

    plt.colorbar(im, ax=axes, pad=.02, aspect=40, extend='both', label=f'{"Re" if func == np.real else "Im"}' + r'[$P(k) / P_N(k)$] (unitless)')

# %%
def plot_SNR_hist(to_hist, theory='laplace', leg_title='All Bands, $|\\tau| > 1000$ ns', bins=None,
                  denom_label='$P_N$'):
    '''Plots the histogram of power spectrum SNR values (both real and imaginary) and compares them to a theoretical distribution.'''

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if bins is None:
        bins = np.linspace(-15,15,200)        

    for ax, func, c in zip(axes, [np.real, np.imag], ['C0', 'C1']):
        ax.hist(func(to_hist), bins=bins, density=True, color=c, edgecolor='k', linewidth=.1, 
                label=f'{"Re" if func == np.real else "Im"}[$P(k)$] / {denom_label}')
        ax.set_yscale('log')

        if theory == 'laplace':
            b = 2**-.5
            laplace = np.exp(-np.abs(bins) / b) / 2 / b
            ax.plot(bins, laplace, 'k--', label='Laplace Distribution')
        elif theory == 'gauss':
            gauss = np.exp(-bins**2/2) / np.sqrt(2*np.pi)
            ax.plot(bins, gauss, 'k--', label='Gaussian Distribution')

        ax.legend(title=leg_title)
        ax.set_xlabel('SNR')
        ax.set_ylabel('Density')
        ax.set_ylim([10**np.floor(np.log10(1 / len(to_hist))), 1])

        text = f'Observed Mean: {np.nanmean(func(to_hist)).real:.3f}'
        text += f'\nObserved Median: {np.nanmedian(func(to_hist)).real:.3f}'
        if theory == 'laplace':
            text += f'\nObserved/Expected Std: {np.nanstd(func(to_hist)) / 2**.5 / b:.3f}'
            text += f'\nObserved/Expected MAD: {np.nanmedian(np.abs(func(to_hist) - np.nanmedian(func(to_hist)))) / b / np.log(2):.3f}'
        elif theory == 'gauss':
            text += f'\nObserved/Expected Std: {np.nanstd(func(to_hist)):.3f}'
            text += f'\nObserved/Expected MAD: {np.nanmedian(np.abs(func(to_hist) - np.nanmedian(func(to_hist)))) / (2**.5 * special.erfinv(.5)):.3f}'

        ax.set_title(text)

# %%
def plot_tavg_pspec():
    '''This plots the time-averaged power spectrum over the whole range of LSTs, including 2 sigma errors'''
    
    fig, axes = plt.subplots(int(np.ceil(len(bands) / 2)), 2, figsize=(18, 12), sharex=True, sharey=True, gridspec_kw={'wspace': .03, 'hspace': .0}, dpi=100)
    for spw, (ax, band, band_slice) in enumerate(zip(np.ravel(axes), bands, band_slices)):

        key = (spw, (ANTPAIR, ANTPAIR), ('pI', 'pI'))

        pk_avg = np.squeeze(uvp_avg_all.get_data(key).real)
        delays = uvp_avg_all.get_dlys(key[0]) * 1e9
        P_N = np.squeeze(uvp_avg_all.get_stats('P_N', key))
        P_SN = np.squeeze(uvp_avg_all.get_stats('P_SN', key))        
        ax.errorbar(delays, pk_avg, marker='o', ls='', yerr=2*P_SN, label='$Re[P(k)]$ with 2$\sigma$ $P_{SN}$ errors')
        ax.plot(delays, P_N, 'k--', label='$P_{N}$')
        ax.plot(delays, P_SN, 'k-', label='$P_{SN}$')
        ax.set_yscale('log')    
        ax.set_xlim([-2500, 2500])
        ax.set_ylim([1e3, 1e14])
        ax.set_xlabel('Delay (ns)')
        ax.tick_params(axis='x', direction='in')
        for multiple in [1, -1]:
            ax.axvline(multiple * dly_filter_half_widths[0] * 1e9, ls='--', color='k', lw=.5, label=(r'Filtering $\tau_{max}$' if multiple == 1 else None))
            ax.axvline(multiple * inpaint_filter_half_widths[0] * 1e9, ls=':', color='k', lw=.5, label=(r'Inpainting $\tau_{max}$' if multiple == 1 else None))
        if spw % 2 == 0:
            ax.set_ylabel('Re[$P(k)$]\n(mK$^2$ $h^{-3}$ Mpc$^3$)')    

        ax.text(.02, .93, f'Band {spw + 1}:\n{band[0]}—{band[1]} MHz', transform=ax.transAxes, fontsize=12,
                va='top', ha='left', bbox=dict(boxstyle='round', facecolor='w', alpha=0.8))
    
    handles, labels = np.ravel(axes)[0].get_legend_handles_labels()    
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, .92), ncol=len(labels))
    
    plt.tight_layout()

# %% [markdown]
# ### *Figure 8: Interleave-Averaged Power Spectra (Pseudo-Stokes I, Q, U, & V) vs. LST*

# %%
plot_Pk_vs_LST(pol='pI')
plot_Pk_vs_LST(pol='pQ')
plot_Pk_vs_LST(pol='pU')
plot_Pk_vs_LST(pol='pV')

# %% [markdown]
# ### *Figure 9: Interleave-Averaged Power Spectrum SNR vs. LST (Real and Imaginary for pI)*

# %%
plot_Pk_SNR_vs_LST(func=np.real)
plot_Pk_SNR_vs_LST(func=np.imag)

# %% [markdown]
# ### *Figure 10: High Delay Power Spectrum SNR Histograms Before and After Incoherent Averaging*

# %%
plot_SNR_hist(np.array([snr for interleave_band in SNRs for snr in interleave_band]), theory='laplace')
plot_SNR_hist(np.array([snr for interleave_band in tavg_SNRs for snr in interleave_band]), theory='gauss', bins=np.linspace(-10,10,100),
              leg_title='All Bands, Time-Averaged,\n$|\\tau| > 1000$ ns')

# %% [markdown]
# ### *Figure 11: Incoherently Averaged Power Spectrum with Error Bars*

# %%
plot_tavg_pspec()

# %% [markdown]
# ## Save Results

# %%
if SAVE_RESULTS:
    # Create pspec container and write all interleaves to it
    psc = hp.PSpecContainer(OUT_PSPEC_FILE, mode='rw', keep_open=False)
    psc.set_pspec('stokespol', 'interleave_averaged', interleaved_uvp, overwrite=True)

    #Create a file containing the effective delay filtering range
    

    # Create pspec container for time-averaged power spectra
    psc_tavg = hp.PSpecContainer(OUT_TAVG_PSPEC_FILE, mode='rw', keep_open=False)
    psc_tavg.set_pspec('stokespol', 'time_and_interleave_averaged', uvp_avg_all, overwrite=True)
    
    # write ancillary data products directly to header attributes
    for outfile in [OUT_PSPEC_FILE, OUT_TAVG_PSPEC_FILE]:
        with h5py.File(outfile, 'r+') as f:
            f['header'].attrs['dpss_coherent_avg_corrections'] = coherent_avg_correction_factors
            f['header'].attrs['frf_losses'] = [frf_losses[band] for band in bands]

# %% [markdown]
# ## Metadata

# %%
for repo in ['numpy', 'scipy', 'astropy', 'hera_cal', 'hera_qm', 'hera_filters', 'hera_pspec', 'hera_notebook_templates', 'pyuvdata']:
    print(f"{repo}: {importlib.import_module(repo).__version__}")


# %%
print(f'Finished execution in {(time.time() - tstart) / 60:.2f} minutes.')
