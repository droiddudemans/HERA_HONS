#!/bin/bash
set -e

# Run delay_filtering_analysis_and_fitting.ipynb once. It finds the optimal delay-filter configurations on the
# sample baselines, fits them against baseline length, tests the fit, and writes fit_result.json under the
# fit results directory. The notebook reads its options from the toml itself.
# Usage: do_DELAY_FILTERING_ANALYSIS_AND_FITTING.sh <toml_file> <conda_env> <nb_dir> <fit_out_dir>

if [ $# -ne 4 ]; then
    echo "Usage: $0 <toml_file> <conda_env> <nb_dir> <fit_out_dir>" >&2
    exit 1
fi

src_dir="$(cd "$(dirname "$0")" && pwd)"
[ -f "${src_dir}/_common.sh" ] && source "${src_dir}/_common.sh"

toml_file="$1"; conda_env="$2"; nb_dir="$3"; fit_out="$4"
# Activate the conda env if one was given and conda is available.
if [ -n "${conda_env}" ] && command -v conda > /dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${conda_env}"
fi

mkdir -p "${fit_out}"

export TOML_FILE="${toml_file}"
export SBPP_NOTEBOOK="${nb_dir}/single_baseline_postprocessing_and_pspec.ipynb"   # the fit notebook runs this one itself
export MPLBACKEND=Agg
export HDF5_USE_FILE_LOCKING=FALSE

papermill "${nb_dir}/delay_filtering_analysis_and_fitting.ipynb" \
    "${fit_out}/delay_filtering_analysis_and_fitting_out.ipynb"
