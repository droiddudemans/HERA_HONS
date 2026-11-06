#!/bin/bash
set -e

# Run delay_filtering_analysis_and_fitting.ipynb once. It finds the optimal delay-filter configurations on the
# sample baselines, fits them against baseline length, tests the fit, and writes fit_result.json under
# FIT_RESULTS_OUT (see hons_fit_result_file). The per-baseline jobs read that file.
# Usage: do_DELAY_FILTERING_ANALYSIS_AND_FITTING.sh <toml_file> <conda_env> <nb_template_dir>

if [ $# -ne 3 ]; then
    echo "Usage: $0 <toml_file> <conda_env> <nb_template_dir>" >&2
    exit 1
fi

src_dir="$(cd "$(dirname "$0")" && pwd)"
[ -f "${src_dir}/_common.sh" ] && source "${src_dir}/_common.sh"
source "${src_dir}/_hons_common.sh"

toml_file="$1"; conda_env="$2"; nb_dir="$3"
hons_activate_env "${conda_env}"

fit_out=$(hons_toml_get "${toml_file}" POSTPROCESS_AND_PSPEC_OPTS FIT_RESULTS_OUT)
fit_json=$(hons_fit_result_file "${toml_file}")
mkdir -p "${fit_out}"

export TOML_FILE="${toml_file}"
export SBPP_NOTEBOOK="${nb_dir}/single_baseline_postprocessing_and_pspec.ipynb"   # the fit notebook calls this one
export MPLBACKEND=Agg
export HDF5_USE_FILE_LOCKING=FALSE

# A stale fit from a previous run must not satisfy the check below.
rm -f "${fit_json}"

papermill "${nb_dir}/delay_filtering_analysis_and_fitting.ipynb" \
    "${fit_out}/delay_filtering_analysis_and_fitting_out.ipynb"

if [ ! -f "${fit_json}" ]; then
    echo "Fit notebook finished but ${fit_json} was not written" >&2
    exit 1
fi
echo "Found ${fit_json}"
