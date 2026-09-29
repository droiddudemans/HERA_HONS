import os
import subprocess
import sys
from pathlib import Path

import plotting
import toml


# ============================================================================
# CONSTANTS
# ============================================================================

EV_CUTOFF_TO_TEST = [
    1e-12,
    1e-10,
    1e-8,
    1e-6,
    1e-4,
    1e-2,
    1e-1
]

EV_CUTOFF_TO_PLOT = [
    1e-12,
    1e-10,
    1e-8,
    1e-6
]

IS_LOG_SCALE = True
IS_INPAINT_HERE = False
IS_REMOVE_FLAGS = False

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
    Construct all validation input/output directories.
    """
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
            / "eigenvalue_test"
        ),
    }


def create_output_directories(directories: dict[str, Path]) -> None:
    """
    Create all directories required by the processing pipeline.
    """
    validation_dir = directories["validation"]
    output_dir = directories["output"]

    directories_to_create = [
        validation_dir,
        directories["eor_foregrounds"],
        directories["eor_only"],
        output_dir,
        output_dir / "plots",
        output_dir / "eor_foregrounds_sum_pspec",
        output_dir / "eor_only_pspec",
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


def process_no_delay_pspec(
    input_file: Path,
    output_dir: Path,
    eigenvalue_cutoff: float,
) -> None:
    """
    Generate a PSPEC without delay filtering if it does not already exist.
    """
    pspec_file = output_dir / f"{input_file.stem}.tavg.pspec.h5"

    if pspec_file.is_file():
        print(f"Skipping existing no-delay pspec: {pspec_file.name}")
        return

    print(f"Processing file without delay filter: {input_file.name}")

    run_postprocessing(
        input_file=input_file,
        delay_filter=False,
        output_dir=output_dir,
        eigenvalue_cutoff=eigenvalue_cutoff,
    )


def process_delay_filtered_pspec(
    input_file: Path,
    output_dir: Path,
    eigenvalue_cutoff: float,
) -> bool:
    """
    Generate a delay-filtered PSPEC if it does not already exist.

    Returns
    -------
    bool
        True if the PSPEC exists after processing, False if processing failed.
    """
    pspec_file = output_dir / f"{input_file.stem}.tavg.pspec.h5"

    if pspec_file.is_file():
        print(f"Skipping existing delay-filtered pspec: {pspec_file.name}")
        return True

    print(
        f"Processing file with delay filter: {input_file.name} "
        f"(EV cutoff = {eigenvalue_cutoff:g})"
    )

    return run_postprocessing(
        input_file=input_file,
        delay_filter=True,
        output_dir=output_dir,
        eigenvalue_cutoff=eigenvalue_cutoff,
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
# PLOTTING
# ============================================================================

def plot_ev_comparative(
    input_file: Path,
    delay_on_dir: Path,
    off_delay_file_pspec: Path,
    eor_file_pspec: Path,
    output_dir: Path,
    half_width: float,
    ev_cutoff_failures: list[float],
) -> None:
    """
    Generate comparative plots for the selected eigenvalue cutoffs.
    """
    delay_pspec_files = []
    labels = []

    for cutoff in EV_CUTOFF_TO_PLOT:
        if cutoff in ev_cutoff_failures:
            print(
                f"Skipping plot for EV cutoff {cutoff:g}: "
                "PSPEC generation failed."
            )
            continue

        cutoff_dir = delay_on_dir / f"ev_cutoff_{cutoff}"

        pspec_file = cutoff_dir / f"{input_file.stem}.tavg.pspec.h5"

        if not pspec_file.is_file():
            raise FileNotFoundError(
                f"Could not find PSPEC for cutoff {cutoff:g} "
                f"in {cutoff_dir}"
            )

        delay_pspec_files.append(pspec_file)
        labels.append(f"EV cutoff = {cutoff:g}")

    plotting.plot_delta2_tau_multiple(
        off_delay_file_pspec,
        delay_pspec_files,
        half_width,
        out_dir = output_dir,
        labels = labels,
        eor_pspec_file = eor_file_pspec,
        log_scale = IS_LOG_SCALE,
        scale_y_axis_detailed = not IS_LOG_SCALE
    )

    plotting.plot_delta2_tau_signed_multiple(
        off_delay_file_pspec,
        delay_pspec_files,
        half_width,
        out_dir = output_dir,
        labels = labels,
        eor_pspec_file = eor_file_pspec,
        log_scale = IS_LOG_SCALE,
        scale_y_axis_detailed = not IS_LOG_SCALE
    )


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
    Process one validation dataset / baseline pair.
    """
    pair, ant1, ant2 = get_baseline_pair(input_file)

    # Skip autocorrelations.
    if ant1 == ant2:
        return

    print(f"\n{'=' * 70}")
    print(f"Processing baseline pair: {pair}")
    print(f"Input file: {input_file.name}")
    print(f"{'=' * 70}")

    output_dir = directories["output"]

    # ------------------------------------------------------------------------
    # Baseline-specific directories
    # ------------------------------------------------------------------------

    baseline_name = f"BLPAIR_{pair}"

    delay_off_dir = (
        output_dir
        / "eor_foregrounds_sum_pspec"
        / "delay_off"
        / baseline_name
    )

    delay_on_dir = (
        output_dir
        / "eor_foregrounds_sum_pspec"
        / "delay_on"
        / baseline_name
    )

    eor_dir = (
        output_dir
        / "eor_only_pspec"
        / baseline_name
    )

    plot_dir_bl = output_dir / "plots" / baseline_name
    plot_dir_bl.mkdir(parents=True, exist_ok=True)

    plot_dir = ""
    if IS_LOG_SCALE:
        plot_dir = plot_dir_bl / "LOG"
    else:
        plot_dir = plot_dir_bl / "NON-LOG"

    for directory in [delay_off_dir, delay_on_dir, eor_dir, plot_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    for directory in [
        delay_off_dir,
        delay_on_dir,
        eor_dir,
        plot_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------
    # Input/output files
    # ------------------------------------------------------------------------

    eor_only_input_file = (
        directories["eor_only"] / input_file.name
    )

    off_delay_file_pspec = (
        delay_off_dir / f"{input_file.stem}.tavg.pspec.h5"
    )

    eor_file_pspec = (
        eor_dir / f"{input_file.stem}.tavg.pspec.h5"
    )

    # ------------------------------------------------------------------------
    # Generate unfiltered PSPECs
    # ------------------------------------------------------------------------

    process_no_delay_pspec(
        input_file=input_file,
        output_dir=delay_off_dir,
        eigenvalue_cutoff=DLY_FILT_EIGENVAL_CUTOFF,
    )

    process_no_delay_pspec(
        input_file=eor_only_input_file,
        output_dir=eor_dir,
        eigenvalue_cutoff=DLY_FILT_EIGENVAL_CUTOFF,
    )

    # ------------------------------------------------------------------------
    # Generate delay-filtered PSPECs
    # ------------------------------------------------------------------------

    ev_cutoff_failures = []
    half_width = None

    for cutoff in EV_CUTOFF_TO_TEST:
        cutoff_dir = delay_on_dir / f"ev_cutoff_{cutoff}"
        cutoff_dir.mkdir(parents=True, exist_ok=True)

        delay_hw_file = (
            cutoff_dir
            / f"{input_file.stem}.tavg.delay_filter_hw.csv"
        )

        success = process_delay_filtered_pspec(
            input_file=input_file,
            output_dir=cutoff_dir,
            eigenvalue_cutoff=cutoff,
        )

        if not success:
            ev_cutoff_failures.append(cutoff)
            continue

        # Read the filter width once, from the first successful cutoff.
        if half_width is None:
            half_width = get_delay_filter_half_width(delay_hw_file)

    # ------------------------------------------------------------------------
    # Plot results
    # ------------------------------------------------------------------------

    if half_width is None:
        print(
            f"No successful delay-filtered PSPECs for {input_file.name}. "
            "Skipping plots."
        )
        return

    plot_ev_comparative(
        input_file=input_file,
        delay_on_dir=delay_on_dir,
        off_delay_file_pspec=off_delay_file_pspec,
        eor_file_pspec=eor_file_pspec,
        output_dir=plot_dir,
        half_width=half_width,
        ev_cutoff_failures=ev_cutoff_failures,
    )


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_data() -> None:
    """
    Process all validation datasets.
    """
    directories = get_output_directories()
    create_output_directories(directories)

    eor_foregrounds_dir = directories["eor_foregrounds"]

    input_files = sorted(eor_foregrounds_dir.glob("*.uvh5"))

    print(f"\nFound {len(input_files)} validation dataset(s).")

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