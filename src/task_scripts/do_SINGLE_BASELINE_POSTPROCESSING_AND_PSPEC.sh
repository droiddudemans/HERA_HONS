#!/bin/bash
set -e

# One baseline file in, two power spectra out:
#   <PSPEC_OUT_DIR>/unfiltered/BLPAIR_<a>_<b>/   no delay filter
#   <PSPEC_OUT_DIR>/filtered/BLPAIR_<a>_<b>/     delay filter with the hw_fract / eigenvalue cutoff predicted from the
#                                                fit written by DELAY_FILTERING_ANALYSIS_AND_FITTING
# Usage: do_SINGLE_BASELINE_POSTPROCESSING_AND_PSPEC.sh <input_file> <toml_file> <toml_section> <conda_env> <nb_template_dir>

if [ $# -ne 5 ]; then
    echo "Usage: $0 <input_file> <toml_file> <toml_section> <conda_env> <nb_template_dir>" >&2
    exit 1
fi

src_dir="$(cd "$(dirname "$0")" && pwd)"
[ -f "${src_dir}/_common.sh" ] && source "${src_dir}/_common.sh"
source "${src_dir}/_hons_common.sh"

input="$1"; toml_file="$2"; conda_env="$4"; nb_dir="$5"
hons_activate_env "${conda_env}"

# makeflow hands over a file name; the data are in the working directory.
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

fit_json=$(hons_fit_result_file "${toml_file}")
if [ ! -f "${fit_json}" ]; then
    echo "Fit results ${fit_json} not found; DELAY_FILTERING_ANALYSIS_AND_FITTING must finish first" >&2
    exit 1
fi

pspec_out=$(hons_toml_get "${toml_file}" POSTPROCESS_AND_PSPEC_OPTS PSPEC_OUT_DIR)
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

# ---- filtered (configuration from the fit) ----
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
