import os
import subprocess
import sys
from pathlib import Path

import plotting
import toml


# ============================================================================
# CONSTANTS
# ============================================================================

# Single eigenvalue cutoff used for the delay filter.

IS_INPAINT_HERE = False
IS_REMOVE_FLAGS = False
IS_LOG_SCALE = False
IS_EXPERIMENTAL = True
IS_PLOT_SINGLE_BASELINE_PSPEC = True

TOML_FILE = os.environ.get(
    "TOML_FILE",
    "/home/Kwuzard/Projects/HERA_HONS/src/HONS/h6c_pspec_11band.toml",
)

SCRIPT_DIR = Path(__file__).resolve().parent
POSTPROCESSING_SCRIPT = SCRIPT_DIR / "single_baseline_postprocessing_and_pspec.py"


# ============================================================================
# CONFIGURATION
# ============================================================================

def load_config(toml_file: str | Path) -> None:
    """
    Load configuration options from the TOML file into the global namespace.

    Configuration is loaded from:
        [GLOBAL_OPTS]
        [POSTPROCESS_AND_PSPEC_OPTS]
    """
    toml_options = toml.load(toml_file)

    for section in ["GLOBAL_OPTS", "POSTPROCESS_AND_PSPEC_OPTS"]:
        if section not in toml_options:
            continue

        print(f"\nLoading config from [{section}] in {toml_file}:")

        for key, value in toml_options[section].items():
            globals()[key.upper()] = value
            print(f"  {key.upper()} = {value!r}")


# ============================================================================
# PATHS
# ============================================================================

def get_output_directories() -> dict[str, Path]:
    """
    Construct all input/output directories.
    """
    if not IS_EXPERIMENTAL:
        validation_dir = (
            Path(__file__).resolve().parent.parent.parent / "validation_data"
        )

        return {
            "validation": validation_dir,
            "eor_foregrounds": validation_dir / "eor_and_foregrounds",
            "eor_only": validation_dir / "eor_only",
            "output": (
                Path(__file__).resolve().parent.parent.parent
                / "output"
                / "simple_bl_output"
                / "validation"
            ),
        }
    else:
        raw_data_dir = (
            Path(__file__).resolve().parent.parent.parent / "raw_data" / "single_baselines_raw_data"
        )

        return {
            "data" : raw_data_dir,
            "output" : (
                Path(__file__).resolve().parent.parent.parent
                / "output"
                / "simple_bl_output"
                / "experiment"
            )
        }


def create_output_directories(directories: dict[str, Path]) -> None:
    """
    Create all directories required by the processing pipeline.
    """
    output_dir = directories["output"]

    directories_to_create = []
    if not IS_EXPERIMENTAL:
        directories_to_create = [
            directories["validation"],
            directories["eor_foregrounds"],
            directories["eor_only"],
            output_dir,
            output_dir / "plots",
            output_dir / "eor_foregrounds_sum_pspec",
            output_dir / "eor_only_pspec",
        ]
    else :
        directories_to_create = [
            directories["data"],
            output_dir,
            output_dir / "plots",
            output_dir / "pspec",
            output_dir / "pspec" / "filtered",
            output_dir / "pspec" / "unfiltered"
        ]
    for directory in directories_to_create:
        directory.mkdir(parents=True, exist_ok=True)


# ============================================================================
# SUBPROCESS / PSPEC PROCESSING
# ============================================================================

def run_postprocessing(
    input_file: Path,
    delay_filter: bool,
    output_dir: Path,
    eigenvalue_cutoff: float,
) -> bool:
    """
    Run the single-baseline postprocessing script.

    Returns
    -------
    bool
        True if processing succeeded, False otherwise.
    """
    command = [
        sys.executable,
        str(POSTPROCESSING_SCRIPT),
        str(input_file),
        str(delay_filter),
        str(output_dir),
        str(DLY_FILT_STANDOFF),
        str(IS_INPAINT_HERE),
        str(eigenvalue_cutoff),
        str(IS_REMOVE_FLAGS),
    ]

    try:
        subprocess.run(command, check=True)
        return True

    except subprocess.CalledProcessError as error:
        print(
            f"Postprocessing failed for {input_file.name} "
            f"(return code: {error.returncode})"
        )
        return False


def process_pspec(
    input_file: Path,
    output_dir: Path,
    delay_filter: bool,
) -> bool:
    """
    Generate a PSPEC (delay-filtered or not) if it does not already exist.

    Returns
    -------
    bool
        True if the PSPEC exists after processing, False if processing failed.
    """
    label = "delay-filtered" if delay_filter else "unfiltered"
    pspec_file = output_dir / f"{input_file.stem}.tavg.pspec.h5"

    print("PSPEC file was: ")
    print(pspec_file)

    if pspec_file.is_file():
        print(f"Skipping existing {label} pspec: {pspec_file.name}")
        return True

    print(f"Processing {label}: {input_file.name}")

    return run_postprocessing(
        input_file=input_file,
        delay_filter=delay_filter,
        output_dir=output_dir,
        eigenvalue_cutoff=DLY_FILT_EIGENVAL_CUTOFF,
    )


# ============================================================================
# DELAY FILTER WIDTH
# ============================================================================

def get_delay_filter_half_width(delay_hw_file: Path) -> float:
    """
    Read the delay-filter half width from the generated CSV file.

    The half width is stored in the second column of the first row.
    """
    with delay_hw_file.open("r") as file:
        first_line = file.readline()

    return float(first_line.split(",")[1])


# ============================================================================
# BASELINE PAIR PROCESSING
# ============================================================================

def get_baseline_pair(input_file: Path) -> tuple[str, str, str]:
    """
    Extract the baseline pair from the input filename.

    Returns
    -------
    tuple
        (pair, antenna_1, antenna_2)
    """
    pair = input_file.name.split(".")[3]
    ant1, ant2 = pair.split("_")

    return pair, ant1, ant2


def process_baseline_pair(
    input_file: Path,
    directories: dict[str, Path],
) -> None:
    """
    Process one dataset / baseline pair.
    """
    pair, ant1, ant2 = get_baseline_pair(input_file)

    # Skip autocorrelations.
    if ant1 == ant2:
        return

    print(f"\n{'=' * 70}")
    print(f"Processing baseline pair: {pair}")
    print(f"Input file: {input_file.name}")
    print(f"EV cutoff: {DLY_FILT_EIGENVAL_CUTOFF:g}")
    print(f"{'=' * 70}")

    output_dir = directories["output"]
    baseline_name = f"BLPAIR_{pair}"

    # ------------------------------------------------------------------------
    # Baseline-specific directories
    # ------------------------------------------------------------------------

    delay_off_dir = None
    delay_on_dir = None
    eor_dir = None
    directories_to_make = []
    if not IS_EXPERIMENTAL:
        delay_off_dir = (
            output_dir / "eor_foregrounds_sum_pspec" / "delay_off" / baseline_name
        )
        delay_on_dir = (
            output_dir
            / "eor_foregrounds_sum_pspec"
            / "delay_on"
            / baseline_name
        )
        eor_dir = output_dir / "eor_only_pspec" / baseline_name
        directories_to_make.append(eor_dir)
    else:
        delay_off_dir = (
            output_dir / "pspec" / "unfiltered" / baseline_name
        )
        delay_on_dir = (
            output_dir / "pspec" / "filtered" / baseline_name
        )

    plot_dir_bl = output_dir / "plots" / baseline_name

    plot_dir_bl.mkdir(parents=True, exist_ok=True)
    plot_dir = ""
    if IS_LOG_SCALE:
        plot_dir = plot_dir_bl / "LOG"
    else:
        plot_dir = plot_dir_bl / "NON-LOG"

    directories_to_make.append(delay_on_dir)
    directories_to_make.append(delay_off_dir)
    directories_to_make.append(plot_dir)
    for directory in directories_to_make:
        directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------
    # Input/output files
    # ------------------------------------------------------------------------

    eor_only_input_file = directories["eor_only"] / input_file.name if not IS_EXPERIMENTAL else None

    off_delay_file_pspec = delay_off_dir / f"{input_file.stem}.tavg.pspec.h5"
    delay_file_pspec = delay_on_dir / f"{input_file.stem}.tavg.pspec.h5"
    eor_file_pspec = eor_dir / f"{input_file.stem}.tavg.pspec.h5" if not IS_EXPERIMENTAL else None
    delay_hw_file = delay_on_dir / f"{input_file.stem}.tavg.delay_filter_hw.csv"

    # ------------------------------------------------------------------------
    # Generate PSPECs
    # ------------------------------------------------------------------------

    if not process_pspec(input_file, delay_off_dir, delay_filter=False):
        print(f"Unfiltered PSPEC failed for {input_file.name}. Skipping plots.")
        return

    if not IS_EXPERIMENTAL and not process_pspec(eor_only_input_file, eor_dir, delay_filter=False):
        print(f"EOR-only PSPEC failed for {input_file.name}. Skipping plots.")
        return

    if not process_pspec(input_file, delay_on_dir, delay_filter=True):
        print(f"Delay-filtered PSPEC failed for {input_file.name}. Skipping plots.")
        return

    # ------------------------------------------------------------------------
    # Plot results
    # ------------------------------------------------------------------------

    half_width = get_delay_filter_half_width(delay_hw_file)

    labels = []
    delay_pspec_files = []
    delay_pspec_files.append(delay_file_pspec)
    labels.append(f"EV cutoff = {DLY_FILT_EIGENVAL_CUTOFF:g}")

    if IS_PLOT_SINGLE_BASELINE_PSPEC:
        plotting.plot_delta2_tau_multiple(
            off_delay_file_pspec,
            delay_pspec_files,
            half_width,
            labels = labels,
            out_dir = plot_dir,
            eor_pspec_file = eor_file_pspec,
            log_scale=IS_LOG_SCALE,
            scale_y_axis_detailed = not IS_LOG_SCALE
        )

        plotting.plot_delta2_tau_signed_multiple(
            off_delay_file_pspec,
            delay_pspec_files,
            half_width,
            labels = labels,
            out_dir = plot_dir,
            eor_pspec_file = eor_file_pspec,
            log_scale = IS_LOG_SCALE,
            scale_y_axis_detailed = not IS_LOG_SCALE
        )


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_data() -> None:
    """
    Process all datasets.
    """
    directories = get_output_directories()
    create_output_directories(directories)

    input_files = None
    if not IS_EXPERIMENTAL :
        input_files = sorted(directories["eor_foregrounds"].glob("*.uvh5"))
    else:
        input_files = sorted(directories["data"].glob("*.uvh5"))

    if not input_files:
        print("ERROR: INPUT FILES COULD NOT BE FOUND.")
        return
    
    print(f"\nFound {len(input_files)} dataset(s).")

    for input_file in input_files:
        process_baseline_pair(
            input_file=input_file,
            directories=directories,
        )


def main() -> None:
    """
    Entry point.
    """
    load_config(TOML_FILE)

    process_data()


if __name__ == "__main__":
    main()
