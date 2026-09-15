import sys
import os
import toml
import numpy as np
import hera_pspec as hp
import matplotlib.pyplot as plt
import subprocess
from pathlib import Path
import helpers
import plotting

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

IS_INPAINT_HERE = False

#-------------------MAIN PROCESSING / TESTING FUNCTIONS-------------------#

def signal_loss_test(
    delay_filtered_sum_pspec, 
    unfiltered_eor_pspec,
    hw_ns,
    output_folder):
    psc_sum = hp.PSpecContainer(delay_filtered_sum_pspec, mode="r")
    psc_eor = hp.PSpecContainer(unfiltered_eor_pspec, mode="r")

    uvp_sum = psc_sum.get_pspec(
        "stokespol",
        "time_and_interleave_averaged"
    )

    uvp_sum_folded = psc_sum.get_pspec(
        "stokespol",
        "time_and_interleave_averaged"
    )

    uvp_sum_folded.fold_spectra()

    frac_out_file_str = output_folder / "Frac_PSPEC_outside_noise_level_post_filter.csv"
    frac_out_file = open(frac_out_file_str, "w")
    frac_out_file.write("Spectral window index, fraction PSPEC > NOISE (Real), fraction PSPEC > NOISE (Imaginary)\n")

    for key in uvp_sum.get_all_keys():
        P_sum_real = np.squeeze(
            uvp_sum_folded.get_data(key).real
        )

        P_sum_imag = np.squeeze(
            uvp_sum_folded.get_data(key).imag
        )

        delays = np.squeeze(
            uvp_sum_folded.get_dlys(key[0])
        )

        delays_ns = delays * 1e9

        PN = np.squeeze(uvp_sum_folded.get_stats("P_N", key))
        delta_2_PN = abs(delays_ns**3 * PN)

        delta_2_sum_real = abs(delays_ns**3 * P_sum_real)
        delta_2_sum_imag = abs(delays_ns**3 * P_sum_imag)

        true_hw = helpers.find_true_hw(uvp_sum, key, hw_ns)
        print(f"The true hw for spectral window {key[0]} was {true_hw}")
        bin_hw_index = np.argmin(np.abs(delays_ns - true_hw))

        R_real = delta_2_sum_real / delta_2_PN
        R_imag = delta_2_sum_imag / delta_2_PN

        count_outside_2sigma_real = 0
        count_outside_2sigma_imag = 0
        n = len(delays_ns) - bin_hw_index
        for i in range(bin_hw_index, len(delays_ns)):
            if R_real[i] > 2.0 : count_outside_2sigma_real += 1
            if R_imag[i] > 2.0 : count_outside_2sigma_imag += 1

        frac_real_outside = count_outside_2sigma_real / n
        frac_imag_outside = count_outside_2sigma_imag / n

        print(f"Spectral window index: {key[0]}")
        print(f"Fraction (real) of pspec outside of noise: {frac_real_outside}")
        print(f"Fraction (complex) of pspec outside of noise: {frac_imag_outside}\n")

        frac_out_file.write(f"{key[0]}, {frac_real_outside}, {frac_imag_outside}\n")

    frac_out_file.close()
    psc_sum._close()
    psc_eor._close()


def process_data():
    is_flags_off = True
    directory_to_data = Path(__file__).parent.parent.parent / "raw_data" / "single_baselines_raw_data"
    directory_to_output = Path(__file__).parent.parent.parent / "output" / "signal_destruction_test"
    directory_comparative_plots = directory_to_output / "comparative_plots"
    pspec_outside_noise = directory_to_output / "pspec_outside_noise"

    dir_eor_foregrounds_pspec_out = directory_to_output / "eor_foregrounds_pspec"
    dir_eor_only_pspec_out = directory_to_output / "eor_only_pspec"

    dir_off_delay = dir_eor_foregrounds_pspec_out / "delay_off"
    dir_on_delay = dir_eor_foregrounds_pspec_out / "delay_on"

    directory_to_data.mkdir(parents = True, exist_ok = True)
    directory_to_output.mkdir(parents = True, exist_ok = True)
    directory_comparative_plots.mkdir(parents = True, exist_ok = True)
    pspec_outside_noise.mkdir(parents = True, exist_ok = True)

    dir_eor_foregrounds_pspec_out.mkdir(parents = True, exist_ok = True)
    dir_eor_only_pspec_out.mkdir(parents = True, exist_ok = True)

    bl_nominal_hw = []
    num_amplified = []

    for input_file in directory_to_data.glob("*.uvh5"):
        #Goes over all validation baseline pairs with boosted eor and foregrounds.
        pair = input_file.name.split(".")[3]
        bl_pair_folder_name = "BLPAIR_" + pair
        ant1, ant2 = pair.split("_")

        if ant1 == ant2:
            continue  # Skip autocorrelations

        #Make a directory (if it doesn't exist) for the baseline pair under investigation
        bl_pair_folder_delay_off = dir_off_delay / bl_pair_folder_name
        bl_pair_folder_delay_on = dir_on_delay / bl_pair_folder_name
        bl_pair_comparative_plots_folder = directory_comparative_plots / bl_pair_folder_name
        bl_pair_pspec_outside_noise = pspec_outside_noise / bl_pair_folder_name

        bl_pair_folder_delay_off.mkdir(parents = True, exist_ok = True)
        bl_pair_folder_delay_on.mkdir(parents = True, exist_ok = True)
        bl_pair_comparative_plots_folder.mkdir(parents = True, exist_ok = True)
        bl_pair_pspec_outside_noise.mkdir(parents = True, exist_ok = True)

        off_delay_file_pspec = bl_pair_folder_delay_off / f"{input_file.stem}.tavg.pspec.h5"
        on_delay_file_pspec = bl_pair_folder_delay_on / f"{input_file.stem}.tavg.pspec.h5"
        delay_hw = bl_pair_folder_delay_on / f"{input_file.stem}.tavg.delay_filter_hw.csv"

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


        #Create PSPEC for delay filtering (foregrounds + eor)
        if not on_delay_file_pspec.is_file():
            print(f"Processing file with delay filter : {input_file.name}")
            try:
                subprocess.run([
                    sys.executable,
                    "single_baseline_postprocessing_and_pspec.py",
                    str(input_file),
                    "True",
                    str(bl_pair_folder_delay_on.absolute()),
                    str(DLY_FILT_MIN_DLY),
                    str(IS_INPAINT_HERE),
                    str(DLY_FILT_EIGENVAL_CUTOFF),
                    str(is_flags_off)
                ], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Exception in subprocess. Failed with error: {e.returncode}")
        else:
            print(f"Skipping existing delay-filtered pspec: {on_delay_file_pspec.name}")

        hw = 0
        print("Got to before z analysis")
        with open(delay_hw, "r") as file:
            line = file.readline()
            hw = float(line.split(',')[1]) #Seond argument in csv file of first line is the half width of the delay filter.
        #hw in nanoseconds btw

        bl_nominal_hw.append(hw)
        num_amplified.append(helpers.count_num_amplified_bl(off_delay_file_pspec, on_delay_file_pspec))

        plotting.plot_delta2_tau(
            off_delay_file_pspec,
            on_delay_file_pspec, 
            bl_pair_comparative_plots_folder, 
            hw,
        )

        plotting.plot_delta2_tau_signed(
            off_delay_file_pspec,
            on_delay_file_pspec,
            bl_pair_comparative_plots_folder,
            hw
        )

        signal_loss_test(
            on_delay_file_pspec, 
            off_delay_file_pspec, 
            hw,
            bl_pair_pspec_outside_noise
        )
    
    fig_amp_count_bl = plt.figure("amp_count_fig_bl", figsize = (10, 4), dpi = 300)
    plt.scatter(bl_nominal_hw, num_amplified, marker = 's')
    plt.xlabel("Nominal baseline delay filter half-width.")
    plt.ylabel("COUNT(amplified bins)")
    plt.title(f"Number of TOTAL amplified PSPEC bins over all windows as delay filter half-width varies.")
    plt.savefig(directory_to_output / "num_amplified_bins_bl.png")

    plt.show()
    plt.close(fig_amp_count_bl)


def main():
    process_data()


if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("Usage: uv run signal_destruction_test.py\n")
        print("The program tests whether for any delay, the temperature is below")
        print("the boosted EOR signal using validation datasets.")
    main()