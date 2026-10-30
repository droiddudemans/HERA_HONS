#!/usr/bin/env bash
set -euo pipefail

out=/home/Kwuzard/Projects/HERA_HONS/output/bulk_processing_out
mkdir -p "$out" "$out/summary_out/filtered" "$out/summary_out/unfiltered"

# 1. Fit on the small subset
# -config is in TOML file.
papermill delay_filtering_analysis_and_fitting.ipynb.ipynb "$out/delay_filtering_analysis_and_fitting_out.ipynb"

# 2. Batch: 800 (or however many) baselines, filtered and unfiltered
# -config is in the TOML file.
papermill run_batch.ipynb "$out/run_batch_out.ipynb" \
    -p BATCH_DATA_DIR /home/Kwuzard/Projects/HERA_HONS/data/bulk_visibilities \
    -p BATCH_OUT_DIR "$out/pspec_out"

# 3. Merge each set
pspecs="$out/pspec_out/pspecs_and_horizons/pspec_out"
./do_MERGE_SINGLE_BASELINE_FILES.sh "$pspecs/filtered"   "$out/pspec_out/filtered_merged"
./do_MERGE_SINGLE_BASELINE_FILES.sh "$pspecs/unfiltered" "$out/pspec_out/unfiltered_merged"

# 4. Summary and signal-loss test
papermill power_spectrum_summary.ipynb "$out/summary_out_filtered.ipynb" \
    -p RESULTS_DIR "$out/summary_out/filtered" \
    -p TAVG_PSPEC_FILE "$out/pspec_out/filtered_merged/baselines_merged.tavg.pspec.h5"

papermill power_spectrum_summary.ipynb "$out/summary_out_unfiltered.ipynb" \
    -p RESULTS_DIR "$out/summary_out/unfiltered" \
    -p TAVG_PSPEC_FILE "$out/pspec_out/unfiltered_merged/baselines_merged.tavg.pspec.h5"