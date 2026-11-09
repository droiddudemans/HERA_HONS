#!/bin/bash
set -e

# One baseline file in, two power spectra out:
#   <pspec_out_dir>/unfiltered/BLPAIR_<a>_<b>/   no delay filter
#   <pspec_out_dir>/filtered/BLPAIR_<a>_<b>/     delay filter with the configuration predicted from the stored fit
# Usage: do_SINGLE_BASELINE_POSTPROCESSING_AND_PSPEC.sh <input_file> <toml_file> <conda_env> <nb_dir> <pspec_out_dir>

if [ $# -ne 5 ]; then
    echo "Usage: $0 <input_file> <toml_file> <conda_env> <nb_dir> <pspec_out_dir>" >&2
    exit 1
fi

src_dir="$(cd "$(dirname "$0")" && pwd)"
[ -f "${src_dir}/_common.sh" ] && source "${src_dir}/_common.sh"

input="$1"; toml_file="$2"; conda_env="$3"; nb_dir="$4"; pspec_out="$5"
# Activate the conda env if one was given and conda is available.
if [ -n "${conda_env}" ] && command -v conda > /dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${conda_env}"
fi

if [ ! -f "${input}" ]; then
    echo "Input file ${input} not found in $(pwd)" >&2
    exit 1
fi
input=$(readlink -f "${input}")

pair=$(basename "${input}" | grep -oE '[0-9]+_[0-9]+' | head -1)   # same rule the notebook uses for ANTPAIR
if [ -z "${pair}" ]; then
    echo "Could not find an antenna pair in ${input}" >&2
    exit 1
fi

nb="${nb_dir}/single_baseline_postprocessing_and_pspec.ipynb"
export TOML_FILE="${toml_file}"
export MPLBACKEND=Agg
export HDF5_USE_FILE_LOCKING=FALSE

# ---- unfiltered ----
out="${pspec_out}/unfiltered/BLPAIR_${pair}"
mkdir -p "${out}"
papermill "${nb}" "${out}/sbpp_output.ipynb" \
    -r TOML_FILE "${toml_file}" -r INPUT_FILE "${input}" -r OUTPUT_DIR "${out}" \
    -p PERFORM_DLY_FILT False -p USE_FIT_RESULTS False

# ---- filtered (the notebook predicts the configuration from the stored fit and fails if there is none) ----
out="${pspec_out}/filtered/BLPAIR_${pair}"
mkdir -p "${out}"
papermill "${nb}" "${out}/sbpp_output.ipynb" \
    -r TOML_FILE "${toml_file}" -r INPUT_FILE "${input}" -r OUTPUT_DIR "${out}" -r HW_OUT_DIR "${out}" \
    -p PERFORM_DLY_FILT True -p USE_FIT_RESULTS True

stem=$(basename "${input}"); stem="${stem%.*}"
for mode in unfiltered filtered; do
    for suffix in pspec.h5 tavg.pspec.h5; do
        f="${pspec_out}/${mode}/BLPAIR_${pair}/${stem}.${suffix}"
        [ -f "${f}" ] || { echo "${f} not produced" >&2; exit 1; }
    done
done
echo "Done: BLPAIR_${pair} (filtered + unfiltered)"
