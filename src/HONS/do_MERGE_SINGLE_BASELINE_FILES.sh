#!/bin/bash
set -e

# Merge single-baseline power spectra into one file.
# Expects <input_dir>/BLPAIR_*/ folders that contain the pspec files.
# Usage: merge_baselines.sh <input_dir> <output_dir>

if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_dir> <output_dir>" >&2
    exit 1
fi

src_dir="$(dirname "$0")"
source "${src_dir}/_common.sh"

in_dir=$(cd "$1" && pwd)
mkdir -p "$2"
out_dir=$(cd "$2" && pwd)

merge() {
    local pattern="$1" names="$2" outpath="$3"

    # Fail early with a clear message if nothing matches the pattern
    if ! compgen -G "${pattern}" > /dev/null; then
        echo "No files match ${pattern}" >&2
        exit 1
    fi

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
merge "${in_dir}/BLPAIR_*/*.sum.pspec.h5" interleave_averaged "${out_dir}/baselines_merged"

# Time-averaged pspecs
merge "${in_dir}/BLPAIR_*/*.sum.tavg.pspec.h5" time_and_interleave_averaged "${out_dir}/baselines_merged.tavg"