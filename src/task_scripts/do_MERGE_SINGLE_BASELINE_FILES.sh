#!/bin/bash
set -e

# Merge the single-baseline power spectra into one filtered and one unfiltered file.
# Reads <PSPEC_OUT_DIR>/{filtered,unfiltered}/BLPAIR_*/ (written by SINGLE_BASELINE_POSTPROCESSING_AND_PSPEC) and writes
#   <PSPEC_OUT_DIR>/merged/filtered/baselines_merged{,.tavg}.pspec.h5
#   <PSPEC_OUT_DIR>/merged/unfiltered/baselines_merged{,.tavg}.pspec.h5
# Usage: do_MERGE_SINGLE_BASELINE_FILES.sh <toml_file> <conda_env>

if [ $# -ne 2 ]; then
    echo "Usage: $0 <toml_file> <conda_env>" >&2
    exit 1
fi

src_dir="$(cd "$(dirname "$0")" && pwd)"
[ -f "${src_dir}/_common.sh" ] && source "${src_dir}/_common.sh"
source "${src_dir}/_hons_common.sh"

toml_file="$1"; conda_env="$2"
hons_activate_env "${conda_env}"

sec=POSTPROCESS_AND_PSPEC_OPTS
pspec_out=$(hons_toml_get "${toml_file}" "${sec}" PSPEC_OUT_DIR)
expected=$(hons_toml_get "${toml_file}" "${sec}" N_EXPECTED_BASELINES 0)

merge() {
    local pattern="$1" names="$2" outpath="$3"

    # Fail early with a clear message if nothing matches the pattern
    if ! compgen -G "${pattern}" > /dev/null; then
        echo "No files match ${pattern}" >&2
        exit 1
    fi

    # Refuse to silently merge a partial set of baselines
    local n
    n=$(compgen -G "${pattern}" | wc -l)
    echo "Merging ${n} files matching ${pattern}"
    if [ "${expected}" -gt 0 ] && [ "${n}" -ne "${expected}" ]; then
        echo "Expected ${expected} files (N_EXPECTED_BASELINES) but found ${n} for ${pattern}" >&2
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

for mode in filtered unfiltered; do
    in_dir="${pspec_out}/${mode}"
    out_dir="${pspec_out}/merged/${mode}"
    mkdir -p "${out_dir}"

    # Non-time-averaged pspecs
    merge "${in_dir}/BLPAIR_*/*.sum.pspec.h5" interleave_averaged "${out_dir}/baselines_merged"

    # Time-averaged pspecs
    merge "${in_dir}/BLPAIR_*/*.sum.tavg.pspec.h5" time_and_interleave_averaged "${out_dir}/baselines_merged.tavg"
done
