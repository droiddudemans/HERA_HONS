#!/bin/bash
set -e

# Run signal_loss_analysis.ipynb on the filtered / unfiltered results. Writes the loss tables (csv) and the
# executed notebook to the signal-loss output directory. Bands, wedge buffer etc. are read from the toml by the notebook.
# Needs the merged pspecs (MERGE_SINGLE_BASELINE_FILES) and the per-band summary files (POWER_SPECTRUM_SUMMARY).
# Usage: do_SIGNAL_LOSS_ANALYSIS.sh <toml_file> <conda_env> <nb_dir> <pspec_out_dir> <summary_out_dir> <signal_loss_out_dir>

if [ $# -ne 6 ]; then
    echo "Usage: $0 <toml_file> <conda_env> <nb_dir> <pspec_out_dir> <summary_out_dir> <signal_loss_out_dir>" >&2
    exit 1
fi

src_dir="$(cd "$(dirname "$0")" && pwd)"
[ -f "${src_dir}/_common.sh" ] && source "${src_dir}/_common.sh"

toml_file="$1"; conda_env="$2"; nb_dir="$3"; pspec_out="$4"; summary="$5"; out="$6"
# Activate the conda env if one was given and conda is available.
if [ -n "${conda_env}" ] && command -v conda > /dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${conda_env}"
fi

unfilt_merged="${pspec_out}/merged/unfiltered/baselines_merged.tavg.pspec.h5"
filt_merged="${pspec_out}/merged/filtered/baselines_merged.tavg.pspec.h5"
for f in "${unfilt_merged}" "${filt_merged}"; do
    [ -f "${f}" ] || { echo "Missing merged pspec ${f}" >&2; exit 1; }
done
for mode in unfiltered filtered; do
    compgen -G "${summary}/${mode}/${mode}.deltasq.1d.spw*.npz" > /dev/null || { echo "No ${mode} summary files in ${summary}/${mode}; POWER_SPECTRUM_SUMMARY must finish first" >&2; exit 1; }
done

mkdir -p "${out}"
export TOML_FILE="${toml_file}"
export MPLBACKEND=Agg

papermill "${nb_dir}/signal_loss_analysis.ipynb" "${out}/signal_loss_analysis_out.ipynb" \
    -r TOML_FILE "${toml_file}" -r OUT_DIR "${out}" \
    -r UNFILT_DIR "${summary}/unfiltered" -r FILT_DIR "${summary}/filtered" \
    -r PSPEC_OUT_DIR "${pspec_out}" \
    -r UNFILT_MERGED_PSPEC "${unfilt_merged}" -r FILT_MERGED_PSPEC "${filt_merged}"

for f in signal_loss_spherical.csv signal_loss_cylindrical_per_band.csv signal_loss_cylindrical_per_baseline.csv; do
    [ -f "${out}/${f}" ] || { echo "${out}/${f} not produced" >&2; exit 1; }
done
echo "Signal-loss tables written to ${out}"
