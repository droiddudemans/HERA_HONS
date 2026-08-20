import sys
import os
import toml
import numpy as np
import hera_pspec as hp
import matplotlib.pyplot as plt
import subprocess
from pathlib import Path


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

#Plots delta^2(tau)
def plot_delta2_tau(
    delay_pspec_file,
    eor_pspec_file,
    out_dir,
    delay_hw,
):
    """
    Plot qualitative

        Delta^2(tau) ~ tau^3 P(tau)

    comparing delay-filtered EOR + foregrounds against EOR only.

    delay_hw is in ns.
    """

    psc_delay = hp.PSpecContainer(delay_pspec_file, mode="r")
    psc_eor = hp.PSpecContainer(eor_pspec_file, mode="r")

    out_dir.mkdir(parents=True, exist_ok=True)

    uvp_delay = psc_delay.get_pspec(
        "stokespol",
        "time_and_interleave_averaged"
    )

    uvp_eor = psc_eor.get_pspec(
        "stokespol",
        "time_and_interleave_averaged"
    )

    print(f"Nominal delay half-width: {delay_hw} ns")

    for key in uvp_delay.get_all_keys():

        P_fg = np.squeeze(uvp_delay.get_data(key).real)
        P_eor = np.squeeze(uvp_eor.get_data(key).real)

        # Delay axis
        delays = np.squeeze(uvp_delay.get_dlys(key[0]))

        # ------------------------------------------------------------
        # Delta^2(tau) ~ tau^3 P(tau)
        #
        # delays is in seconds here.
        # ------------------------------------------------------------

        tau_ns = np.abs(delays) * 1e9

        delta2_fg = tau_ns**3 * P_fg
        delta2_eor = tau_ns**3 * P_eor

        # ------------------------------------------------------------
        # Log-log requires:
        #
        #   tau > 0
        #   Delta^2 > 0
        # ------------------------------------------------------------

        valid_fg = (
            (tau_ns > 0)
            & np.isfinite(delta2_fg)
            & (delta2_fg > 0)
        )

        valid_eor = (
            (tau_ns > 0)
            & np.isfinite(delta2_eor)
            & (delta2_eor > 0)
        )

        # ------------------------------------------------------------
        # Plot
        # ------------------------------------------------------------

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.loglog(
            tau_ns[valid_fg],
            delta2_fg[valid_fg],
            lw=2,
            label="EOR + foregrounds"
        )

        ax.loglog(
            tau_ns[valid_eor],
            delta2_eor[valid_eor],
            lw=2,
            label="EOR only"
        )

        # Delay-filter half width
        ax.axvline(
            delay_hw,
            color="k",
            linestyle="--",
            linewidth=1.5,
            label=fr"$\tau_\mathrm{{hw}}={delay_hw:.1f}$ ns"
        )

        ax.set_xlabel(r"$|\tau|$ [ns]")

        ax.set_ylabel(r"$\Delta^2(\tau) \propto \tau^3 P(\tau)$")

        ax.set_title(f"Delay spectrum — SPW {key[0]}")

        ax.grid(True, which = "both", alpha=0.25)

        ax.legend()

        fig.tight_layout()

        output_file = (out_dir / f"delta2_tau_spw_{key[0]}.png")

        fig.savefig(output_file, dpi = 300, bbox_inches="tight")

        plt.show()
        plt.close(fig)

        print(f"Saved: {output_file}")


def process_data():
    directory_to_validation_datasets = Path(__file__).parent.parent.parent / "validation_data"
    directory_to_eor_and_foregrounds = directory_to_validation_datasets / "eor_and_foregrounds"
    directory_to_eor_only_datasets = directory_to_validation_datasets / "eor_only"
    directory_to_validation_output = Path(__file__).parent.parent.parent / "validation_out"
    directory_comparative_plots = directory_to_validation_output / "comparative_plots"

    dir_eor_foregrounds_pspec_out = directory_to_validation_output / "eor_foregrounds_pspec"
    dir_eor_only_pspec_out = directory_to_validation_output / "eor_only_pspec"

    dir_off_delay = dir_eor_foregrounds_pspec_out / "delay_off"
    dir_on_delay = dir_eor_foregrounds_pspec_out / "delay_on"

    directory_to_validation_datasets.mkdir(parents = True, exist_ok = True)
    directory_to_validation_output.mkdir(parents = True, exist_ok = True)
    directory_comparative_plots.mkdir(parents = True, exist_ok = True)

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
        bl_pair_comparative_plots_folder = directory_comparative_plots / bl_pair_folder_name

        bl_pair_folder_delay_off.mkdir(parents = True, exist_ok = True)
        bl_pair_folder_delay_on.mkdir(parents = True, exist_ok = True)
        bl_pair_folder_eor.mkdir(parents = True, exist_ok = True)
        bl_pair_comparative_plots_folder.mkdir(parents = True, exist_ok = True)

        eor_only_input_file = directory_to_eor_only_datasets / input_file.name

        off_delay_file_pspec = bl_pair_folder_delay_off / f"{input_file.stem}.tavg.pspec.h5"
        eor_file_pspec = bl_pair_folder_eor / f"{input_file.stem}.tavg.pspec.h5"
        on_delay_file_pspec = bl_pair_folder_delay_on / f"{input_file.stem}.tavg.pspec.h5"
        delay_hw = bl_pair_folder_delay_on / f"{input_file.stem}_cutoff_{DLY_FILT_MIN_DLY}.tavg.delay_filter_hw.csv"

        #Create PSPEC for no delay (foregrounds + eor)
        if not off_delay_file_pspec.is_file():
            print(f"Processing file without delay filter : {input_file.name}")
            try:
                subprocess.run([
                    sys.executable,
                    "single_baseline_postprocessing_and_pspec.py",
                    str(input_file),
                    "False",
                    str(dir_off_delay.absolute()),
                    str(DLY_FILT_MIN_DLY),
                    str(IS_INPAINT_HERE)
                ], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Exception in subprocess. Failed with error: {e.returncode}")
        else:
            print(f"Skipping existing no-delay pspec: {off_delay_file_pspec.name}")

        #Create PSPEC for no delay (eor only)
        if not eor_file_pspec.is_file():
            print(f"Processing file (eor only) without delay filter : {eor_file_pspec.name}")
            try:
                subprocess.run([
                    sys.executable,
                    "single_baseline_postprocessing_and_pspec.py",
                    str(eor_only_input_file),
                    "False",
                    str(eor_file_pspec.absolute()),
                    str(DLY_FILT_MIN_DLY),
                    str(IS_INPAINT_HERE)
                ], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Exception in subprocess. Failed with error: {e.returncode}")
        else:
            print(f"Skipping existing no-delay (eor only) pspec: {eor_file_pspec.name}")

        #Create PSPEC for delay filtering (foregrounds + eor)
        if not on_delay_file_pspec.is_file():
            print(f"Processing file with delay filter : {input_file.name}")
            try:
                subprocess.run([
                    sys.executable,
                    "single_baseline_postprocessing_and_pspec.py",
                    str(input_file),
                    "True",
                    str(dir_on_delay.absolute()),
                    str(DLY_FILT_MIN_DLY),
                    str(IS_INPAINT_HERE)
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

        plot_delta2_tau(on_delay_file_pspec, eor_file_pspec, bl_pair_comparative_plots_folder, hw)


def main():
    process_data()


if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("Usage: uv run signal_destruction_test.py\n")
        print("The program tests whether for any delay, the temperature is below")
        print("the boosted EOR signal using validation datasets.")
    main()