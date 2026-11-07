#!/bin/bash
set -e

# Run signal_loss_analysis.ipynb on the merged filtered / unfiltered results. Writes the loss tables (csv) and the
# executed notebook to SIGNAL_LOSS_DIR.
#
# Input it needs, besides the merged pspecs from MERGE_SINGLE_BASELINE_FILES: the per-band
#   <SUMMARY_DIR>/{unfiltered,filtered}/{unfiltered,filtered}.deltasq.{1d,2d}.spwNN.npz
# written by POWER_SPECTRUM_SUMMARY.
# Usage: do_SIGNAL_LOSS_ANALYSIS.sh <toml_file> <conda_env> <nb_template_dir>

if [ $# -ne 3 ]; then
    echo "Usage: $0 <toml_file> <conda_env> <nb_template_dir>" >&2
    exit 1
fi

src_dir="$(cd "$(dirname "$0")" && pwd)"
[ -f "${src_dir}/_common.sh" ] && source "${src_dir}/_common.sh"
source "${src_dir}/_hons_common.sh"

toml_file="$1"; conda_env="$2"; nb_dir="$3"
hons_activate_env "${conda_env}"

sec=POSTPROCESS_AND_PSPEC_OPTS
pspec_out=$(hons_toml_get "${toml_file}" "${sec}" PSPEC_OUT_DIR)
summary=$(hons_toml_get "${toml_file}" "${sec}" SUMMARY_DIR)
out=$(hons_toml_get "${toml_file}" "${sec}" SIGNAL_LOSS_DIR)
bands=$(hons_toml_get "${toml_file}" "${sec}" SUMMARY_BANDS)
buffer=$(hons_toml_get "${toml_file}" "${sec}" SUMMARY_WEDGE_BUFFER_NS)

unfilt_merged="${pspec_out}/merged/unfiltered/baselines_merged.tavg.pspec.h5"
filt_merged="${pspec_out}/merged/filtered/baselines_merged.tavg.pspec.h5"

# Fail early, with the full list of what is missing.
missing=0
for f in "${unfilt_merged}" "${filt_merged}"; do
    [ -f "${f}" ] || { echo "Missing merged pspec ${f}" >&2; missing=1; }
done
for mode in unfiltered filtered; do
    for b in ${bands//,/ }; do
        nn=$(printf "%02d" "${b}")
        for d in 1d 2d; do
            f="${summary}/${mode}/${mode}.deltasq.${d}.spw${nn}.npz"
            [ -f "${f}" ] || { echo "Missing summary output ${f}" >&2; missing=1; }
        done
    done
done
if [ "${missing}" -ne 0 ]; then
    echo "POWER_SPECTRUM_SUMMARY must finish first (outputs into ${summary})." >&2
    exit 1
fi

mkdir -p "${out}"
export TOML_FILE="${toml_file}"
export MPLBACKEND=Agg

papermill "${nb_dir}/signal_loss_analysis.ipynb" "${out}/signal_loss_analysis_out.ipynb" \
    -r TOML_FILE "${toml_file}" \
    -r OUT_DIR "${out}" \
    -r UNFILT_DIR "${summary}/unfiltered" -r FILT_DIR "${summary}/filtered" \
    -r PSPEC_OUT_DIR "${pspec_out}" \
    -r UNFILT_MERGED_PSPEC "${unfilt_merged}" -r FILT_MERGED_PSPEC "${filt_merged}" \
    -r BANDS "${bands}" -p WEDGE_BUFFER_NS "${buffer}"

for f in signal_loss_spherical.csv signal_loss_cylindrical_per_band.csv signal_loss_cylindrical_per_baseline.csv; do
    [ -f "${out}/${f}" ] || { echo "${out}/${f} not produced" >&2; exit 1; }
done
echo "Signal-loss tables written to ${out}"
