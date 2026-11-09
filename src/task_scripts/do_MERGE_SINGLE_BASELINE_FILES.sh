#!/bin/bash
set -e

# Merge the single-baseline power spectra into one filtered and one unfiltered file.
# Reads <pspec_out_dir>/{filtered,unfiltered}/BLPAIR_*/ and writes
#   <pspec_out_dir>/merged/filtered/baselines_merged{,.tavg}.pspec.h5
#   <pspec_out_dir>/merged/unfiltered/baselines_merged{,.tavg}.pspec.h5
# Usage: do_MERGE_SINGLE_BASELINE_FILES.sh <pspec_out_dir> <conda_env>

if [ $# -ne 2 ]; then
    echo "Usage: $0 <pspec_out_dir> <conda_env>" >&2
    exit 1
fi

src_dir="$(cd "$(dirname "$0")" && pwd)"
[ -f "${src_dir}/_common.sh" ] && source "${src_dir}/_common.sh"

pspec_out="$1"; conda_env="$2"
# Activate the conda env if one was given and conda is available.
if [ -n "${conda_env}" ] && command -v conda > /dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${conda_env}"
fi

count() { compgen -G "$1" | wc -l; }

merge() {
    local pattern="$1" names="$2" outpath="$3"

    # Fail early with a clear message if nothing matches the pattern
    if ! compgen -G "${pattern}" > /dev/null; then
        echo "No files match ${pattern}" >&2
        exit 1
    fi
    echo "Merging $(count "${pattern}") files matching ${pattern}"

    pspec fast-merge-baselines --pattern "${pattern}" --group stokespol \
        --names "${names}" --outpath "${outpath}" --extras frf_losses

    if [ -f "${outpath}.pspec.h5" ]; then
        echo "Found ${outpath}.pspec.h5"
    else
        echo "${outpath}.pspec.h5 not produced" >&2
        exit 1
    fi
}

# The filtered and unfiltered merges must contain the same baselines.
for suffix in sum.pspec.h5 sum.tavg.pspec.h5; do
    n_f=$(count "${pspec_out}/filtered/BLPAIR_*/*.${suffix}")
    n_u=$(count "${pspec_out}/unfiltered/BLPAIR_*/*.${suffix}")
    if [ "${n_f}" -ne "${n_u}" ]; then
        echo "Found ${n_f} filtered but ${n_u} unfiltered *.${suffix} files" >&2
        exit 1
    fi
done

for mode in filtered unfiltered; do
    in_dir="${pspec_out}/${mode}"
    out_dir="${pspec_out}/merged/${mode}"
    mkdir -p "${out_dir}"

    # Non-time-averaged pspecs
    merge "${in_dir}/BLPAIR_*/*.sum.pspec.h5" interleave_averaged "${out_dir}/baselines_merged"

    # Time-averaged pspecs
    merge "${in_dir}/BLPAIR_*/*.sum.tavg.pspec.h5" time_and_interleave_averaged "${out_dir}/baselines_merged.tavg"
done
