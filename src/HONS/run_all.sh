#!/usr/bin/env bash
set -euo pipefail

# 1. Fit on the small subset
# -config is in TOML file.
papermill delay_filter_analysis.ipynb out/delay_filter_analysis_out.ipynb \
    -p OUTPUT_DIR /home/Kwuzard/Projects/HERA_HONS/output/signal_loss_test_out \
    -p DATA_DIR /home/Kwuzard/Projects/HERA_HONS/data \
    -p IS_EOR_ONLY True \
    -p IS_EXPERIMENTAL_TESTS False

# 2. Batch: 800 (or however many) baselines, filtered and unfiltered
# -config is in the TOML file.
out=/home/Kwuzard/Projects/HERA_HONS/output/bulk_processing_out
python run_batch.py \
    -p BATCH_DATA_DIR /home/Kwuzard/Projects/HERA_HONS/data/bulk_visibilities \
    -p BATCH_OUT_DIR $out/pspec_out

# 3. Merge each set
./merge_baselines.sh $out/pspec_out/filtered_pspecs   $out/pspec_out/filtered_merged
./merge_baselines.sh $out/pspec_out/unfiltered_pspecs $out/pspec_out/unfiltered_merged

# 4. Summary and signal-loss test
papermill power_spectrum_summary.ipynb $out/summary_out.ipynb \
    -p RESULTS_DIR $out/summary_out/filtered \
    -p TAVG_PSPEC_FILE $out/pspec_out/filtered_merged/baselines_merged.tavg.pspec.h5

papermill power_spectrum_summary.ipynb $out/summary_out.ipynb \
    -p RESULTS_DIR $out/summary_out/unfiltered \
    -p TAVG_PSPEC_FILE $out/pspec_out/unfiltered_merged/baselines_merged.tavg.pspec.h5