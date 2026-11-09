#!/bin/bash
set -e

# Run power_spectrum_summary.ipynb on the merged unfiltered pspec and then on the merged filtered pspec.
# Output (CASENAME is the mode name, which signal_loss_analysis.ipynb expects):
#   <summary_out_dir>/unfiltered/unfiltered.deltasq.{1d,2d}.spwNN.{npz,h5}, figures
#   <summary_out_dir>/filtered/filtered.deltasq.{1d,2d}.spwNN.{npz,h5}, figures
# Bands, wedge buffer etc. are read from the toml by the notebook.
# Usage: do_POWER_SPECTRUM_SUMMARY.sh <toml_file> <conda_env> <nb_dir> <pspec_out_dir> <summary_out_dir>

if [ $# -ne 5 ]; then
    echo "Usage: $0 <toml_file> <conda_env> <nb_dir> <pspec_out_dir> <summary_out_dir>" >&2
    exit 1
fi

src_dir="$(cd "$(dirname "$0")" && pwd)"
[ -f "${src_dir}/_common.sh" ] && source "${src_dir}/_common.sh"

toml_file="$1"; conda_env="$2"; nb_dir="$3"; pspec_out="$4"; summary="$5"
# Activate the conda env if one was given and conda is available.
if [ -n "${conda_env}" ] && command -v conda > /dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${conda_env}"
fi

export TOML_FILE="${toml_file}"
export MPLBACKEND=Agg
export HDF5_USE_FILE_LOCKING=FALSE

for mode in unfiltered filtered; do
    tavg="${pspec_out}/merged/${mode}/baselines_merged.tavg.pspec.h5"
    frf="${tavg%.pspec.h5}.frf_losses.pkl"      # written next to the merge by fast-merge-baselines --extras frf_losses
    for f in "${tavg}" "${frf}"; do
        [ -f "${f}" ] || { echo "Missing ${f}" >&2; exit 1; }
    done

    out="${summary}/${mode}"
    mkdir -p "${out}"
    echo "Summarising ${mode}: ${tavg}"
    papermill "${nb_dir}/power_spectrum_summary.ipynb" "${out}/power_spectrum_summary_out.ipynb" \
        -r TOML_FILE "${toml_file}" -r TAVG_PSPEC_FILE "${tavg}" -r RESULTS_DIR "${out}" \
        -r CASENAME "${mode}" -r FRF_SIGNAL_LOSS_FILE "${frf}"

    for d in 1d 2d; do
        compgen -G "${out}/${mode}.deltasq.${d}.spw*.npz" > /dev/null || { echo "No ${mode}.deltasq.${d}.spw*.npz in ${out}" >&2; exit 1; }
    done
done
echo "Summaries written to ${summary}"
