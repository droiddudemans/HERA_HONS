from pathlib import Path
import sys
import os
import toml
import subprocess
import plotting
from pathlib import Path

MAX_DELAY = 1000
MIN_DELAY = 100
DECR_DELAY = 100
IS_INPAINT_HERE = False

#LOADING TOML CONFIGS
TOML_FILE = os.environ.get('TOML_FILE',
        '/home/Kwuzard/Projects/HERA_HONS/src/HONS/h6c_pspec_11band.toml')

# Load configuration from the toml and inject into globals.
toml_options = toml.load(TOML_FILE)
for toml_section in ['GLOBAL_OPTS', 'POSTPROCESS_AND_PSPEC_OPTS']:
    if toml_section not in toml_options:
        continue
    print(f'\nLoading config from [{toml_section}] in {TOML_FILE}:')
    for key, val in toml_options[toml_section].items():
        globals()[key.upper()] = val
        print(f'  {key.upper()} = {val!r}')

#------------------------------------#
#---------PLOTTING FUNCTIONS---------#
#------------------------------------#

def plot_delay_comparative(
    input_file,
    bl_pair_folder_delay_on,
    off_delay_file_pspec,
    eor_file_pspec,
    output_dir,
    delay_failures
):
    DELAY_CUTOFFS_TO_PLOT = list(range(MAX_DELAY, MIN_DELAY - 1, -DECR_DELAY))

    delay_pspec_files = []
    labels = []
    hws = []

    for standoff in DELAY_CUTOFFS_TO_PLOT:
        if standoff in delay_failures:
            continue

        folder = bl_pair_folder_delay_on / f"standoff_{standoff}_ns"

        matching_files = list(folder.glob(f"{input_file.stem}.tavg.pspec.h5"))

        if not matching_files:
            raise FileNotFoundError(
                f"Could not find PSPEC for delay standoff {standoff} in {folder}"
            )

        delay_pspec_files.append(matching_files[0])
        labels.append(f"Delay standoff = {standoff} ns")
        hws.append(standoff)  # the requested standoff is the half-width for this curve

    plotting.plot_delta2_tau_multiple(
        off_delay_file_pspec,
        eor_file_pspec,
        delay_pspec_files,
        output_dir,
        hws,
        labels
    )

    plotting.plot_delta2_tau_signed_multiple(
        off_delay_file_pspec,
        eor_file_pspec,
        delay_pspec_files,
        output_dir,
        hws,
        labels
    )

#--------------------------------------#
#---------PROCESSING FUNCTIONS---------#
#--------------------------------------#

def process_data_delay_standoff():
    directory_to_validation_datasets = Path(__file__).parent.parent.parent / "validation_data"
    directory_to_eor_and_foregrounds = directory_to_validation_datasets / "eor_and_foregrounds"
    directory_to_eor_only_datasets = directory_to_validation_datasets / "eor_only"
    directory_to_validation_output = Path(__file__).parent.parent.parent / "output" / "delay_standoff_test"
    directory_plots = directory_to_validation_output / "plots"

    dir_eor_foregrounds_pspec_out = directory_to_validation_output / "eor_foregrounds_sum_pspec"
    dir_eor_only_pspec_out = directory_to_validation_output / "eor_only_pspec"

    dir_off_delay = dir_eor_foregrounds_pspec_out / "delay_off"
    dir_on_delay = dir_eor_foregrounds_pspec_out / "delay_on"

    directory_to_validation_datasets.mkdir(parents=True, exist_ok=True)
    directory_to_validation_output.mkdir(parents=True, exist_ok=True)
    directory_plots.mkdir(parents=True, exist_ok=True)

    dir_eor_foregrounds_pspec_out.mkdir(parents=True, exist_ok=True)
    dir_eor_only_pspec_out.mkdir(parents=True, exist_ok=True)

    directory_to_eor_and_foregrounds.mkdir(parents=True, exist_ok=True)
    directory_to_eor_only_datasets.mkdir(parents=True, exist_ok=True)

    for input_file in directory_to_eor_and_foregrounds.glob("*.uvh5"):
        # Goes over all validation baseline pairs with boosted eor and foregrounds.
        pair = input_file.name.split(".")[3]
        bl_pair_folder_name = "BLPAIR_" + pair
        ant1, ant2 = pair.split("_")

        if ant1 == ant2:
            continue  # Skip autocorrelations

        bl_pair_folder_delay_off = dir_off_delay / bl_pair_folder_name
        bl_pair_folder_delay_on = dir_on_delay / bl_pair_folder_name
        bl_pair_folder_eor = dir_eor_only_pspec_out / bl_pair_folder_name
        bl_pair_comparative_plots_folder = directory_plots / bl_pair_folder_name

        bl_pair_folder_delay_off.mkdir(parents=True, exist_ok=True)
        bl_pair_folder_delay_on.mkdir(parents=True, exist_ok=True)
        bl_pair_folder_eor.mkdir(parents=True, exist_ok=True)
        bl_pair_comparative_plots_folder.mkdir(parents=True, exist_ok=True)

        eor_only_input_file = directory_to_eor_only_datasets / input_file.name

        off_delay_file_pspec = bl_pair_folder_delay_off / f"{input_file.stem}.tavg.pspec.h5"
        eor_file_pspec = bl_pair_folder_eor / f"{input_file.stem}.tavg.pspec.h5"

        delay_standoff_failures = []

        # Create PSPEC for no delay (foregrounds + eor)
        if not off_delay_file_pspec.is_file():
            print(f"Processing file without delay filter : {input_file.name}")
            try:
                subprocess.run([
                    sys.executable,
                    "single_baseline_postprocessing_and_pspec.py",
                    str(input_file),
                    "False",
                    str(bl_pair_folder_delay_off.absolute()),
                    str(DLY_FILT_STANDOFF),
                    str(IS_INPAINT_HERE),
                    str(DLY_FILT_EIGENVAL_CUTOFF),
                    str(False)
                ], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Exception in subprocess. Failed with error: {e.returncode}")
        else:
            print(f"Skipping existing no-delay pspec: {off_delay_file_pspec.name}")

        # Create PSPEC for no delay (eor only)
        if not eor_file_pspec.is_file():
            print(f"Processing file (eor only) without delay filter : {input_file.name}")
            try:
                subprocess.run([
                    sys.executable,
                    "single_baseline_postprocessing_and_pspec.py",
                    str(eor_only_input_file),
                    "False",
                    str(bl_pair_folder_eor.absolute()),
                    str(DLY_FILT_STANDOFF),
                    str(IS_INPAINT_HERE),
                    str(DLY_FILT_EIGENVAL_CUTOFF),
                    str(False)
                ], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Exception in subprocess. Failed with error: {e.returncode}")
        else:
            print(f"Skipping existing no-delay (eor only) pspec: {eor_file_pspec.name}")

        # Calculate the delay-filtered PSPEC for each delay standoff
        current_delay = MAX_DELAY
        while current_delay >= MIN_DELAY:
            delay_standoff_folder = bl_pair_folder_delay_on / f"standoff_{current_delay}_ns"
            delay_standoff_folder.mkdir(parents=True, exist_ok=True)

            on_delay_file_pspec = delay_standoff_folder / f"{input_file.stem}.tavg.pspec.h5"

            if not on_delay_file_pspec.is_file():
                print(f"Processing file with delay filter (standoff={current_delay} ns) : {input_file.name}")
                try:
                    subprocess.run([
                        sys.executable,
                        "single_baseline_postprocessing_and_pspec.py",
                        str(input_file),
                        "True",
                        str(delay_standoff_folder.absolute()),
                        str(current_delay),
                        str(IS_INPAINT_HERE),
                        str(DLY_FILT_EIGENVAL_CUTOFF),
                        str(False)
                    ], check=True)
                except subprocess.CalledProcessError as e:
                    delay_standoff_failures.append(current_delay)
                    print(f"Exception in subprocess. Failed with error: {e.returncode}")
            else:
                print(f"Skipping existing delay-filtered pspec: {on_delay_file_pspec.name}")

            current_delay -= DECR_DELAY

        plot_delay_comparative(
            input_file,
            bl_pair_folder_delay_on,
            off_delay_file_pspec,
            eor_file_pspec,
            bl_pair_comparative_plots_folder,
            delay_standoff_failures
        )

if __name__ == "__main__":
    process_data_delay_standoff()