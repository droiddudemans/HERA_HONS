import sys
import os
import toml
import subprocess
import plotting
from pathlib import Path

# GLOBAL CONSTANTS
EV_CUTOFF_TO_TEST = [
    1e-12,
    1e-10,
    1e-8,
    1e-6,
    1e-4,
    1e-2,
    0.05,
    0.1,
    0.2,
    0.3,
    0.5,
    0.7,
    0.8,
    0.9,
    0.95,
    0.99,
    0.999,
    1.0 - 1e-4,
    1.0 - 1e-6,
    1.0 - 1e-8,
    1.0 - 1e-10,
    1.0 - 1e-12
]


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

# GLOBAL SETTINGS
IS_INPAINT_HERE = False

def plot_ev_comparative(
    input_file,
    bl_pair_folder_delay_on,
    off_delay_file_pspec,
    eor_file_pspec,
    output_dir,
    hw,
    ev_failrues
):
    labels = [
        f"EV cutoff: {cutoff}"
        for cutoff in EV_CUTOFF_TO_TEST
    ]


    EV_CUTOFF_TO_PLOT = [
        1e-12,
        1e-10,
        1e-8,
        1e-6,
        1e-2,
        0.1,
        0.5,
        0.7,
        0.9,
        0.99,
        1.0 - 1e-4,
    ]

    delay_pspec_files = []
    labels = []

    for cutoff in EV_CUTOFF_TO_PLOT:
        if cutoff in ev_failrues: continue
        folder = bl_pair_folder_delay_on / f"ev_cutoff_{cutoff}"

        matching_files = list(
            folder.glob(f"{input_file.stem}.tavg.pspec.h5")
        )

        if not matching_files:
            raise FileNotFoundError(
                f"Could not find PSPEC for cutoff {cutoff} in {folder}"
            )

        delay_pspec_files.append(matching_files[0])
        labels.append(f"EV cutoff = {cutoff:g}")
        
    plotting.plot_delta2_tau_multiple(
        off_delay_file_pspec,
        eor_file_pspec,
        delay_pspec_files,
        output_dir,
        hw,
        labels
    )
    
    plotting.plot_delta2_tau_signed_multiple(
        off_delay_file_pspec,
        eor_file_pspec,
        delay_pspec_files,
        output_dir,
        hw,
        labels
    )


def process_data():
    directory_to_validation_datasets = Path(__file__).parent.parent.parent / "validation_data"
    directory_to_eor_and_foregrounds = directory_to_validation_datasets / "eor_and_foregrounds"
    directory_to_eor_only_datasets = directory_to_validation_datasets / "eor_only"
    directory_to_validation_output = Path(__file__).parent.parent.parent / "output" / "eigenvalue_test"
    directory_plots = directory_to_validation_output / "plots"

    dir_eor_foregrounds_pspec_out = directory_to_validation_output / "eor_foregrounds_sum_pspec"
    dir_eor_only_pspec_out = directory_to_validation_output / "eor_only_pspec"

    dir_off_delay = dir_eor_foregrounds_pspec_out / "delay_off"
    dir_on_delay = dir_eor_foregrounds_pspec_out / "delay_on"

    directory_to_validation_datasets.mkdir(parents = True, exist_ok = True)
    directory_to_validation_output.mkdir(parents = True, exist_ok = True)
    directory_plots.mkdir(parents = True, exist_ok = True)

    dir_eor_foregrounds_pspec_out.mkdir(parents = True, exist_ok = True)
    dir_eor_only_pspec_out.mkdir(parents = True, exist_ok = True)

    directory_to_eor_and_foregrounds.mkdir(parents = True, exist_ok = True)
    directory_to_eor_only_datasets.mkdir(parents = True, exist_ok = True)

    for input_file in directory_to_eor_and_foregrounds.glob("*.uvh5"):
        #Goes over all validation baseline pairs with boosted eor and foregrounds.
        pair = input_file.name.split(".")[3]
        bl_pair_folder_name = "BLPAIR_" + pair
        ant1, ant2 = pair.split("_")

        if ant1 == ant2:
            continue  # Skip autocorrelations

        #Make a directory (if it doesn't exist) for the baseline pair under investigation
        bl_pair_folder_delay_off = dir_off_delay / bl_pair_folder_name
        bl_pair_folder_delay_on = dir_on_delay / bl_pair_folder_name
        bl_pair_folder_eor = dir_eor_only_pspec_out / bl_pair_folder_name
        bl_pair_comparative_plots_folder = directory_plots / bl_pair_folder_name

        bl_pair_folder_delay_off.mkdir(parents = True, exist_ok = True)
        bl_pair_folder_delay_on.mkdir(parents = True, exist_ok = True)
        bl_pair_folder_eor.mkdir(parents = True, exist_ok = True)
        bl_pair_comparative_plots_folder.mkdir(parents = True, exist_ok = True)

        eor_only_input_file = directory_to_eor_only_datasets / input_file.name

        # These files will be output after subprocesses run
        off_delay_file_pspec = bl_pair_folder_delay_off / f"{input_file.stem}.tavg.pspec.h5"
        eor_file_pspec = bl_pair_folder_eor / f"{input_file.stem}.tavg.pspec.h5"


        ev_cutoff_failures = []

        #Create PSPEC for no delay (foregrounds + eor)
        if not off_delay_file_pspec.is_file():
            print(f"Processing file without delay filter : {input_file.name}")
            try:
                subprocess.run([
                    sys.executable,
                    "single_baseline_postprocessing_and_pspec.py",
                    str(input_file),
                    "False",
                    str(bl_pair_folder_delay_off.absolute()),
                    str(DLY_FILT_MIN_DLY),
                    str(IS_INPAINT_HERE),
                    str(DLY_FILT_EIGENVAL_CUTOFF),
                    str(False)
                ], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Exception in subprocess. Failed with error: {e.returncode}")
        else:
            print(f"Skipping existing no-delay pspec: {off_delay_file_pspec.name}")

        #Create PSPEC for no delay (eor only)
        if not eor_file_pspec.is_file():
            print(f"Processing file (eor only) without delay filter : {input_file.name}")
            try:
                subprocess.run([
                    sys.executable,
                    "single_baseline_postprocessing_and_pspec.py",
                    str(eor_only_input_file),
                    "False",
                    str(bl_pair_folder_eor.absolute()),
                    str(DLY_FILT_MIN_DLY),
                    str(IS_INPAINT_HERE),
                    str(DLY_FILT_EIGENVAL_CUTOFF),
                    str(False)
                ], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Exception in subprocess. Failed with error: {e.returncode}")
        else:
            print(f"Skipping existing no-delay (eor only) pspec: {eor_file_pspec.name}")

        # Calculate the delay-filtered PSPEC for each eigenvalue cutoff
        hw = 0
        for i in range(0, len(EV_CUTOFF_TO_TEST)):
            ev_cutoff_folder = bl_pair_folder_delay_on / f"ev_cutoff_{EV_CUTOFF_TO_TEST[i]}"
            ev_cutoff_folder.mkdir(parents = True, exist_ok = True)

            on_delay_file_pspec = ev_cutoff_folder / f"{input_file.stem}.tavg.pspec.h5"
            delay_hw = ev_cutoff_folder / f"{input_file.stem}.tavg.delay_filter_hw.csv"

            #Create PSPEC for delay filtering (foregrounds + eor)
            if not on_delay_file_pspec.is_file():
                print(f"Processing file with delay filter : {input_file.name}")
                try:
                    subprocess.run([
                        sys.executable,
                        "single_baseline_postprocessing_and_pspec.py",
                        str(input_file),
                        "True",
                        str(ev_cutoff_folder.absolute()),
                        str(DLY_FILT_MIN_DLY),
                        str(IS_INPAINT_HERE),
                        str(EV_CUTOFF_TO_TEST[i]),
                        str(False)
                    ], check=True)
                except subprocess.CalledProcessError as e:
                    ev_cutoff_failures.append(EV_CUTOFF_TO_TEST[i])
                    print(f"Exception in subprocess. Failed with error: {e.returncode}")
            else:
                print(f"Skipping existing delay-filtered pspec: {on_delay_file_pspec.name}")


            if hw == 0:
                with open(delay_hw, "r") as file:
                    line = file.readline()
                    hw = float(line.split(',')[1]) #Seond argument in csv file of first line is the half width of the delay filter.
                #hw in nanoseconds btw

        plot_ev_comparative(input_file, bl_pair_folder_delay_on, off_delay_file_pspec, eor_file_pspec, bl_pair_comparative_plots_folder, hw, ev_cutoff_failures)

def main():
    process_data()

if __name__ == "__main__":
    main()