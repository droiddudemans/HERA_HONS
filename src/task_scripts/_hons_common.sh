#!/bin/bash
# Shared helpers for the HONS makeflow task scripts (sourcing this).

# Activate the conda env if one was given and conda is around.
hons_activate_env() {
    local env_name="$1"
    if [ -n "${env_name}" ] && command -v conda > /dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate "${env_name}"
    fi
}

# hons_toml_get <toml_file> <section> <key> [default]
# Prints the value (path, number, bool as true/false), or the default, or fails if there is no default.
hons_toml_get() {
    python - "$1" "$2" "$3" "${4-__NODEFAULT__}" <<'PY'
import sys
toml_file, section, key, default = sys.argv[1:5]
try:
    import tomllib
    with open(toml_file, "rb") as f:
        cfg = tomllib.load(f)
except ImportError:
    import toml
    cfg = toml.load(toml_file)
try:
    val = cfg[section][key]
except KeyError:
    if default == "__NODEFAULT__":
        sys.exit(f"[{section}] {key} not found in {toml_file}")
    val = default
print(str(val).lower() if isinstance(val, bool) else val)
PY
}

# Directory that holds the fit written by delay_filtering_analysis_and_fitting.ipynb.
# Must match fit_result_path() in single_baseline_postprocessing_and_pspec.ipynb.
hons_fit_result_file() {
    local toml_file="$1" sec="POSTPROCESS_AND_PSPEC_OPTS"
    local out experimental eor_only base
    out=$(hons_toml_get "${toml_file}" "${sec}" FIT_RESULTS_OUT) || return 1
    experimental=$(hons_toml_get "${toml_file}" "${sec}" FIT_ON_EXPERIMENTAL) || return 1
    eor_only=$(hons_toml_get "${toml_file}" "${sec}" FIT_ON_EOR_ONLY) || return 1
    if [ "${experimental}" = "true" ]; then
        base="${out}/experiment"
    elif [ "${eor_only}" = "true" ]; then
        base="${out}/validation/eor_only"
    else
        base="${out}/validation/eor_and_foregrounds"
    fi
    echo "${base}/fit_testing/fit_results/fit_result.json"
}
