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

def main():
    output_dir_delay_filter.mkdir(parents=True, exist_ok=True)
    output_dir_no_delay_filter.mkdir(parents=True, exist_ok=True)

    psc_yes = hp.PSpecContainer(
        "zen.LST.baseline.0_2.sum.FR0filt.tavg.pspec.h5",
        mode="r"
    )

    psc_no = hp.PSpecContainer("zen.LST.baseline.0_2.sum.FR0filt.tavg.pspec_nodelay.h5")

    uvp_yes = psc_yes.get_pspec(
        "stokespol",
        "time_and_interleave_averaged"
    )

    uvp_no = psc_no.get_pspec(
        "stokespol",
        "time_and_interleave_averaged"
    )

    spw = 7

    delay = uvp_yes.get_dlys(spw)

    print(delay[:5])           # First few delays
    print(np.diff(delay[:5]))  # First few spacings

    bin_width = np.diff(delay)

    print("Unique spacings:", np.unique(bin_width))
    print("Bin size:", bin_width[1] * pow(10, 9))

    power_no = uvp_no.data_array[spw][0, :, 0]
    power_yes = uvp_yes.data_array[spw][0, :, 0]

    delay = uvp_no.get_dlys(spw)

    ratio = power_yes / power_no

    plt.plot(delay, ratio.real)
    plt.xlabel("Delay (s)")
    plt.ylabel("Power ratio")
    plt.grid(True)

    plt.savefig("power_ratio.png", dpi=300, bbox_inches="tight")
        
    return
    # Run through all baselines with no delay filter
    for input_file in input_dir_raw.glob("*.uvh5"):
        delay = False
        print(f"Processing file without delay filter : {input_file.name}")
        subprocess.run([
            sys.executable,
            "single_baseline_postprocessing_and_pspec.py",
            str(input_file),
            str(delay),
            str(output_dir_no_delay_filter.absolute())
        ], check=True)

        delay = True
        print(f"Processing file with delay filter : {input_file.name}")
        subprocess.run([
            sys.executable,
            "single_baseline_postprocessing_and_pspec.py",
            str(input_file),
            str(delay),
            str(output_dir_no_delay_filter.absolute())
        ], check=True)
        no_delay_file_name = str(output_dir_no_delay_filter) + str(input_file) + ".pspec.h5"
        delay_file_name = str(output_dir_delay_filter) + str(input_file) + ".pspec.h5"
        psc_no = hp.PSpecContainer(no_delay_file_name, mode="r")
        psc_yes = hp.PSpecContainer(delay_file_name, mode="r")




if __name__ == "__main__":
    main()