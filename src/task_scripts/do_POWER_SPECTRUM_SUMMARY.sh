#!/bin/bash
set -e

# Run power_spectrum_summary.ipynb on the merged unfiltered pspec and then on the merged filtered pspec.
# Output (CASENAME = the mode name, which signal_loss_analysis.ipynb expects):
#   <SUMMARY_DIR>/unfiltered/unfiltered.deltasq.{1d,2d}.spwNN.{npz,h5}, figures
#   <SUMMARY_DIR>/filtered/filtered.deltasq.{1d,2d}.spwNN.{npz,h5}, figures
# Usage: do_POWER_SPECTRUM_SUMMARY.sh <toml_file> <conda_env> <nb_template_dir>

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
bands=$(hons_toml_get "${toml_file}" "${sec}" SUMMARY_BANDS)
buffer=$(hons_toml_get "${toml_file}" "${sec}" SUMMARY_WEDGE_BUFFER_NS)
wf_glob=$(hons_toml_get "${toml_file}" "${sec}" SUMMARY_WINDOW_FUNCTION_FILEGLOB)

# The pspecs only contain the bands in BAND_STR; asking for a higher band number would crash the notebook late.
n_bands=$(python - "${toml_file}" <<'PY'
import sys
try:
    import tomllib
    cfg = tomllib.load(open(sys.argv[1], "rb"))
except ImportError:
    import toml
    cfg = toml.load(sys.argv[1])
print(len(cfg["GLOBAL_OPTS"]["BAND_STR"].split(",")))
PY
)
for b in ${bands//,/ }; do
    if [ "${b}" -lt 1 ] || [ "${b}" -gt "${n_bands}" ]; then
        echo "SUMMARY_BANDS contains ${b}, but BAND_STR defines only ${n_bands} bands" >&2
        exit 1
    fi
done

nb="${nb_dir}/power_spectrum_summary.ipynb"
export MPLBACKEND=Agg
export HDF5_USE_FILE_LOCKING=FALSE

for mode in unfiltered filtered; do
    tavg="${pspec_out}/merged/${mode}/baselines_merged.tavg.pspec.h5"
    frf="${tavg%.pspec.h5}.frf_losses.pkl"      # written next to the merge by fast-merge-baselines --extras frf_losses
    for f in "${tavg}" "${frf}"; do
        [ -f "${f}" ] || { echo "Missing ${f}" >&2; exit 1; }
    done

    out="${summary}/${mode}"
    mkdir -p "${out}"
    echo "Summarising ${mode}: ${tavg}"
    papermill "${nb}" "${out}/power_spectrum_summary_out.ipynb" \
        -r TAVG_PSPEC_FILE "${tavg}" -r RESULTS_DIR "${out}" -r CASENAME "${mode}" \
        -r FRF_SIGNAL_LOSS_FILE "${frf}" -r WINDOW_FUNCTION_FILEGLOB "${wf_glob}" \
        -r BANDS_TO_USE "${bands}" -p WEDGE_BUFFER_NS "${buffer}"

    for b in ${bands//,/ }; do
        nn=$(printf "%02d" "${b}")
        for d in 1d 2d; do
            f="${out}/${mode}.deltasq.${d}.spw${nn}.npz"
            [ -f "${f}" ] || { echo "${f} not produced" >&2; exit 1; }
        done
    done
done
echo "Summaries written to ${summary}"
