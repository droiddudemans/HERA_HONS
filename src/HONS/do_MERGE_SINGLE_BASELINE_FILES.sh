#!/bin/bash
set -e

# Merge single-baseline power spectra into one file.
# Usage: merge_baselines.sh <input_dir> <output_dir>

src_dir="$(dirname "$0")"
source "${src_dir}/_common.sh"

in_dir=$(cd "$1" && pwd)
mkdir -p "$2"
out_dir=$(cd "$2" && pwd)

merge() {
    local pattern="$1" names="$2" outpath="$3"
    pspec fast-merge-baselines --pattern "${pattern}" --group stokespol \
        --names "${names}" --outpath "${outpath}" --extras frf_losses

    if [ -f "${outpath}.pspec.h5" ]; then
        echo "Found ${outpath}.pspec.h5"
    else
        echo "${outpath}.pspec.h5 not produced" >&2
        exit 1
    fi
}

# Non-time-averaged pspecs
merge "${in_dir}/*.sum.pspec.h5" interleave_averaged "${out_dir}/baselines_merged"

# Time-averaged pspecs
merge "${in_dir}/*.sum.tavg.pspec.h5" time_and_interleave_averaged "${out_dir}/baselines_merged.tavg"