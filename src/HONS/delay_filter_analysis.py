from pathlib import Path
import sys
import subprocess
import hera_pspec as hp
import numpy as np
import os
import toml
import matplotlib.pyplot as plt

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

input_dir_raw = Path(__file__).parent.parent.parent / "raw_data" / "single_baselines_raw_data"
output_dir_delay_filter = Path(__file__).parent.parent.parent / "output" / "output_single_baseline_delay_filter"
output_dir_no_delay_filter = Path(__file__).parent.parent.parent / "output" / "output_single_baseline_no_delay_filter"
output_dir_inpaint_test = Path(__file__).parent.parent.parent / "output" / "output_inpaint_test"

def output_power_ratio(no_delay_file_name, delay_file_name, dir_out):
    psc_no = hp.PSpecContainer(no_delay_file_name, mode="r")
    psc_yes = hp.PSpecContainer(delay_file_name, mode="r")

    uvp_yes = psc_yes.get_pspec(
        "stokespol",
        "time_and_interleave_averaged"
    )

    uvp_no = psc_no.get_pspec(
        "stokespol",
        "time_and_interleave_averaged"
    )

    #Record for each spectral window the ratio spectrum in delay space for this particular delay cutoff
    for spw in uvp_yes.spw_array:
        pspec_yes = uvp_yes.data_array[spw][0, :, 0]
        pspec_no = uvp_no.data_array[spw][0, :, 0]

        delays = uvp_yes.get_dlys(spw)
        ratio = pspec_yes / pspec_no
        outfile = dir_out / f"ratio_spectral_window_{spw}.csv"
        np.savetxt(outfile,
            np.column_stack((delays, ratio)),
            delimiter = ",",
            header = "delay_ns, ratio_pspec",
            comments = ""
        )
    psc_no._close()
    psc_yes._close()

def power_ratio_calculation(input_file, delay_cutoff, bl_pair_folder_name):
    dir_output_cutoff = Path(__file__).parent.parent.parent / "output" / "delay_cutoff_variation" / f"{bl_pair_folder_name}" / f"cutoff_{delay_cutoff}_ns"
    no_delay_file_name = output_dir_no_delay_filter / f"{input_file.stem}.tavg.pspec.h5"
    delay_file_name = output_dir_delay_filter / f"{input_file.stem}.tavg.pspec.h5"

    dir_output_cutoff.mkdir(parents=True, exist_ok=True) #Make sure output directory exists
    output_power_ratio(no_delay_file_name, delay_file_name, dir_output_cutoff)


def test_delay_cutoff():
    MAX_DELAY = 1000
    MIN_DELAY = 200
    DECR_DELAY = 100
    IS_INPAINT_HERE = True
    current_delay = MAX_DELAY

    #Loop over delays
    while current_delay >= MIN_DELAY:
        #Loop over baselines
        print(f"Running over all baselines with delay max cutoff : {current_delay}")
        for input_file in input_dir_raw.glob("*.uvh5"):
            delay = False
            pair = input_file.name.split(".")[3]
            bl_pair_folder_name = "BLPAIR_" + pair
            ant1, ant2 = pair.split("_")

            if ant1 == ant2:
                continue  # Skip autocorrelations

            print(f"Processing file without delay filter : {input_file.name}")
            try:
                subprocess.run([
                    sys.executable,
                    "single_baseline_postprocessing_and_pspec.py",
                    str(input_file),
                    str(delay),
                    str(output_dir_no_delay_filter.absolute()),
                    str(current_delay),
                    str(IS_INPAINT_HERE)
                ], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Exception in subproccess. Failed with error: {e.returncode}")
    
            delay = True
            print(f"Processing file with delay filter : {input_file.name}")

            try:
                subprocess.run([
                    sys.executable,
                    "single_baseline_postprocessing_and_pspec.py",
                    str(input_file),
                    str(delay),
                    str(output_dir_delay_filter.absolute()),
                    str(current_delay),
                    str(IS_INPAINT_HERE)
                ], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Exception in subproccess. Failed with error: {e.returncode}")

            output_power_ratio(input_file, current_delay, bl_pair_folder_name)
        current_delay -= DECR_DELAY

def test_inpainting():
    MIN_DELAY_TEST = 150
    IS_INPAINT_HERE = True
    output_inpaint_delay_on = output_dir_inpaint_test / "delay_filter_on"
    output_inpaint_delay_off = output_dir_inpaint_test / "delay_filter_off"
    file_to_test = Path(__file__).parent.parent.parent / "raw_data" / "single_baselines_raw_data" / "zen.LST.baseline.0_3.sum.FR0filt.uvh5"

    is_delay = False
    try:
        subprocess.run([
            sys.executable,
            "single_baseline_postprocessing_and_pspec.py",
            str(file_to_test),
            str(is_delay),
            str(output_inpaint_delay_off.absolute()),
            str(MIN_DELAY_TEST),
            str(IS_INPAINT_HERE)
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Exception in subproccess. Failed with error: {e.returncode}")

    is_delay = True
    try:
        subprocess.run([
            sys.executable,
            "single_baseline_postprocessing_and_pspec.py",
            str(file_to_test),
            str(is_delay),
            str(output_inpaint_delay_on.absolute()),
            str(MIN_DELAY_TEST),
            str(IS_INPAINT_HERE)
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Exception in subproccess. Failed with error: {e.returncode}")
    no_delay_file_name = output_inpaint_delay_off / f"{file_to_test.stem}.tavg.pspec.h5"
    delay_file_name = output_inpaint_delay_on / f"{file_to_test.stem}.tavg.pspec.h5"
    output_power_ratio(no_delay_file_name, delay_file_name, output_dir_inpaint_test)

def main(mode):
    if mode == 0:
        #Test if different minimum delay cutoffs have an effect on the P(after) / P(Before) for delay filtering.
        test_delay_cutoff()
    elif mode == 1:
        #Test whether inpainting decreases fluctuations on the P(After) / P(Before) for delay filtering.
        test_inpainting()

if __name__ == "__main__":
    if len(sys.argv) == 2: main(int(sys.argv[1]))