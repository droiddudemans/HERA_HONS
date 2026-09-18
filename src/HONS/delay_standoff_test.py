import os
import subprocess
import sys
from pathlib import Path

import plotting
import toml


# ============================================================================
# CONSTANTS
# ============================================================================

MAX_DELAY = 1000
MIN_DELAY = 100
DECR_DELAY = 100

IS_INPAINT_HERE = False
IS_LOG_SCALE = False

TOML_FILE = os.environ.get(
    "TOML_FILE",
    "/home/Kwuzard/Projects/HERA_HONS/src/HONS/h6c_pspec_11band.toml",
)

SCRIPT_DIR = Path(__file__).resolve().parent
POSTPROCESSING_SCRIPT = (
    SCRIPT_DIR / "single_baseline_postprocessing_and_pspec.py"
)


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

    for section in [
        "GLOBAL_OPTS",
        "POSTPROCESS_AND_PSPEC_OPTS",
    ]:
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
    Construct the input and output directories used by the validation test.
    """
    project_root = (
        Path(__file__).resolve().parent.parent.parent
    )

    validation_dir = project_root / "validation_data"

    return {
        "validation": validation_dir,
        "eor_foregrounds": validation_dir / "eor_and_foregrounds",
        "eor_only": validation_dir / "eor_only",
        "output": project_root / "output" / "delay_standoff_test",
    }


def create_output_directories(
    directories: dict[str, Path],
) -> None:
    """
    Create the directories required by the processing pipeline.
    """
    output_dir = directories["output"]

    directories_to_create = [
        directories["validation"],
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
    delay_standoff: float,
) -> bool:
    """
    Run the single-baseline postprocessing script.

    Parameters
    ----------
    input_file
        Input UVH5 file.
    delay_filter
        Whether the delay filter should be applied.
    output_dir
        Directory where PSPEC products should be written.
    delay_standoff
        Delay-filter standoff in ns.

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
        str(delay_standoff),
        str(IS_INPAINT_HERE),
        str(DLY_FILT_EIGENVAL_CUTOFF),
        str(False),
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
) -> None:
    """
    Generate an unfiltered PSPEC if it does not already exist.
    """
    pspec_file = output_dir / f"{input_file.stem}.tavg.pspec.h5"

    if pspec_file.is_file():
        print(
            f"Skipping existing no-delay pspec: "
            f"{pspec_file.name}"
        )
        return

    print(
        f"Processing file without delay filter: "
        f"{input_file.name}"
    )

    run_postprocessing(
        input_file=input_file,
        delay_filter=False,
        output_dir=output_dir,
        delay_standoff=DLY_FILT_STANDOFF,
    )


def process_delay_filtered_pspec(
    input_file: Path,
    output_dir: Path,
    delay_standoff: float,
) -> bool:
    """
    Generate a delay-filtered PSPEC if it does not already exist.

    Returns
    -------
    bool
        True if processing succeeded or the PSPEC already exists.
    """
    pspec_file = output_dir / f"{input_file.stem}.tavg.pspec.h5"

    if pspec_file.is_file():
        print(
            f"Skipping existing delay-filtered pspec: "
            f"{pspec_file.name}"
        )
        return True

    print(
        f"Processing file with delay filter "
        f"(standoff={delay_standoff} ns): "
        f"{input_file.name}"
    )

    return run_postprocessing(
        input_file=input_file,
        delay_filter=True,
        output_dir=output_dir,
        delay_standoff=delay_standoff,
    )


# ============================================================================
# PLOTTING
# ============================================================================

def plot_delay_comparative(
    input_file: Path,
    delay_on_dir: Path,
    off_delay_file_pspec: Path,
    eor_file_pspec: Path,
    output_dir: Path,
    delay_failures: list[int],
) -> None:
    """
    Generate comparative plots for all successful delay standoffs.
    """
    delay_standoffs_to_plot = list(
        range(
            MAX_DELAY,
            MIN_DELAY - 1,
            -DECR_DELAY * 2,
        )
    )

    delay_pspec_files = []
    labels = []
    half_widths = []

    for standoff in delay_standoffs_to_plot:
        if standoff in delay_failures:
            print(
                f"Skipping plot for delay standoff "
                f"{standoff} ns: PSPEC generation failed."
            )
            continue

        standoff_dir = (
            delay_on_dir
            / f"standoff_{standoff}_ns"
        )

        pspec_file = (
            standoff_dir
            / f"{input_file.stem}.tavg.pspec.h5"
        )

        if not pspec_file.is_file():
            raise FileNotFoundError(
                f"Could not find PSPEC for delay standoff "
                f"{standoff} ns in {standoff_dir}"
            )

        delay_pspec_files.append(pspec_file)
        labels.append(
            f"Delay standoff = {standoff} ns"
        )

        # The requested delay standoff is the half-width
        # associated with this curve.
        half_widths.append(standoff)

    plotting.plot_delta2_tau_multiple(
        off_delay_file_pspec,
        eor_file_pspec,
        delay_pspec_files,
        output_dir,
        half_widths,
        labels,
        log_scale=IS_LOG_SCALE
    )

    plotting.plot_delta2_tau_signed_multiple(
        off_delay_file_pspec,
        eor_file_pspec,
        delay_pspec_files,
        output_dir,
        half_widths,
        labels,
        log_scale=IS_LOG_SCALE
    )


# ============================================================================
# BASELINE PAIR PROCESSING
# ============================================================================

def get_baseline_pair(
    input_file: Path,
) -> tuple[str, str, str]:
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

    plot_dir = (
        output_dir
        / "plots"
        / baseline_name
    )

    for directory in [
        delay_off_dir,
        delay_on_dir,
        eor_dir,
        plot_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------
    # Input / output files
    # ------------------------------------------------------------------------

    eor_only_input_file = (
        directories["eor_only"]
        / input_file.name
    )

    off_delay_file_pspec = (
        delay_off_dir
        / f"{input_file.stem}.tavg.pspec.h5"
    )

    eor_file_pspec = (
        eor_dir
        / f"{input_file.stem}.tavg.pspec.h5"
    )

    # ------------------------------------------------------------------------
    # Generate unfiltered PSPECs
    # ------------------------------------------------------------------------

    # Foregrounds + EoR
    process_no_delay_pspec(
        input_file=input_file,
        output_dir=delay_off_dir,
    )

    # EoR only
    process_no_delay_pspec(
        input_file=eor_only_input_file,
        output_dir=eor_dir,
    )

    # ------------------------------------------------------------------------
    # Generate delay-filtered PSPECs
    # ------------------------------------------------------------------------

    delay_failures = []

    for standoff in range(
        MAX_DELAY,
        MIN_DELAY - 1,
        -DECR_DELAY,
    ):
        standoff_dir = (
            delay_on_dir
            / f"standoff_{standoff}_ns"
        )

        standoff_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        success = process_delay_filtered_pspec(
            input_file=input_file,
            output_dir=standoff_dir,
            delay_standoff=standoff,
        )

        if not success:
            delay_failures.append(standoff)

    # ------------------------------------------------------------------------
    # Plot results
    # ------------------------------------------------------------------------

    plot_delay_comparative(
        input_file=input_file,
        delay_on_dir=delay_on_dir,
        off_delay_file_pspec=off_delay_file_pspec,
        eor_file_pspec=eor_file_pspec,
        output_dir=plot_dir,
        delay_failures=delay_failures,
    )


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_data_delay_standoff() -> None:
    """
    Process all validation datasets for the delay-standoff test.
    """
    directories = get_output_directories()
    create_output_directories(directories)

    eor_foregrounds_dir = directories["eor_foregrounds"]

    input_files = list(eor_foregrounds_dir.glob("*.uvh5"))


    print(
        f"\nFound {len(input_files)} validation dataset(s)."
    )

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
    process_data_delay_standoff()


if __name__ == "__main__":
    main()