import sys
import os
import toml
import numpy as np
import hera_pspec as hp
import matplotlib as matplot
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


def plot_delta2_tau_signed(
    unfiltered_pspec_file,
    delay_pspec_file,
    eor_pspec_file,
    out_dir,
    delay_hw,
):
    """
    Plot Delta^2(tau) against the signed delay axis.

    Negative and positive delays remain on their respective sides
    of zero, while both are shown on the same axes.
    """

    psc_unfiltered = hp.PSpecContainer(unfiltered_pspec_file, mode="r")
    psc_delay = hp.PSpecContainer(delay_pspec_file, mode="r")
    psc_eor = hp.PSpecContainer(eor_pspec_file, mode="r")

    out_dir.mkdir(parents=True, exist_ok=True)

    uvp_unfiltered = psc_unfiltered.get_pspec(
        "stokespol",
        "time_and_interleave_averaged"
    )

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

        P_unfiltered = np.squeeze(
            uvp_unfiltered.get_data(key).real
        )

        P_fg = np.squeeze(
            uvp_delay.get_data(key).real
        )

        P_eor = np.squeeze(
            uvp_eor.get_data(key).real
        )

        PN = np.squeeze(uvp_delay.get_stats("P_N", key))
        # Delay axis
        delays = np.squeeze(
            uvp_delay.get_dlys(key[0])
        )

        # Convert seconds -> ns
        delays_ns = delays * 1e9

        # ------------------------------------------------------------
        # Delta^2(tau) ~ |tau|^3 P(tau)
        #
        # IMPORTANT:
        # The x-axis retains the SIGN of tau.
        # Only the tau^3 weighting uses |tau|.
        # ------------------------------------------------------------

        delta2_unfiltered = np.abs(delays_ns)**3 * P_unfiltered
        delta2_fg = np.abs(delays_ns)**3 * P_fg
        delta2_eor = np.abs(delays_ns)**3 * P_eor
        delta2_PN = np.abs(delays_ns)**3 * PN

        # ------------------------------------------------------------
        # Finite positive values only
        # ------------------------------------------------------------

        valid_fg = (
            np.isfinite(delta2_fg)
        )

        valid_eor = (
            np.isfinite(delta2_eor)
        )

        delta2_neg_indices = delta2_fg < 0
        tau_neg_pspec = delays_ns[delta2_neg_indices]
        delta2_fg_avg_neg = delta2_fg[delta2_neg_indices]
        # ------------------------------------------------------------
        # Separate negative and positive delays
        #
        # BUT do NOT take abs() of the delay itself.
        # ------------------------------------------------------------

        negative_fg = (
            valid_fg
            & (delays_ns < 0)
        )

        positive_fg = (
            valid_fg
            & (delays_ns > 0)
        )

        negative_unfiltered = (delays_ns < 0)
        positive_unfiltered = (delays_ns > 0)

        negative_PN = (delays_ns < 0)
        positive_PN = (delays_ns > 0)

        negative_eor = (
            valid_eor
            & (delays_ns < 0)
        )

        positive_eor = (
            valid_eor
            & (delays_ns > 0)
        )

        # ------------------------------------------------------------
        # Plot
        # ------------------------------------------------------------

        fig, ax = plt.subplots(figsize=(10, 6))

        # EOR + foregrounds
        ax.plot(
            delays_ns[negative_fg],
            abs(delta2_fg[negative_fg]),
            lw = 2,
            color = "red",
            label = rf"$abs(\Delta^2(\tau))$ for EOR + foregrounds"
        )

        ax.plot(
            delays_ns[positive_fg],
            abs(delta2_fg[positive_fg]),
            lw=2,
            color="red"
        )

        # EOR only
        ax.plot(
            delays_ns[negative_eor],
            delta2_eor[negative_eor],
            lw = 2,
            linestyle = "--",
            color = "blue",
            label = rf"$\Delta^2(\tau)$ for EOR only (folded)"
        )

        ax.plot(
            delays_ns[positive_eor],
            delta2_eor[positive_eor],
            lw = 2,
            linestyle = "--",
            color = "blue"
        )

        # Noise plots
        ax.plot(
            delays_ns[negative_PN],
            delta2_PN[negative_PN],
            lw = 1,
            linestyle = ":",
            color = 'k',
            label = rf"$\Delta^2(\tau)$ for Noise ($P_N$)"
        )

        ax.plot(
            delays_ns[positive_PN],
            delta2_PN[positive_PN],
            lw=1,
            linestyle=":",
            color = 'k',
        )

        #Plot unfiltered
        ax.plot(
            delays_ns[negative_unfiltered],
            abs(delta2_unfiltered[negative_unfiltered]),
            lw = 2,
            color = "orange",
            label = rf"$abs(\Delta^2(\tau))$ for unfiltered EOR + foregrounds"
        )

        ax.plot(
            delays_ns[positive_unfiltered],
            abs(delta2_unfiltered[positive_unfiltered]),
            lw = 2,
            linestyle = "--",
            color = "orange",
        )

        ax.scatter(
            tau_neg_pspec,
            abs(delta2_fg_avg_neg),
            lw=2,
            color="blue",
            label=rf"$\Delta^2(\tau) < 0$ for EOR + foregrounds"
        )

        # ------------------------------------------------------------
        # Delay filter boundaries
        # ------------------------------------------------------------

        ax.axvline(
            -delay_hw,
            color="k",
            linestyle="--",
            linewidth=1.5,
            label=fr"$\pm\tau_\mathrm{{hw}}={delay_hw:.1f}$ ns"
        )

        ax.axvline(
            delay_hw,
            color="k",
            linestyle="--",
            linewidth=1.5
        )

        # Zero-delay reference
        ax.axvline(
            0,
            color="gray",
            linestyle=":",
            linewidth=1
        )

        # ------------------------------------------------------------
        # Labels
        # ------------------------------------------------------------

        ax.set_xlabel(r"$\tau$ [ns]")

        ax.set_ylabel(
            r"$\Delta^2(\tau) \propto |\tau|^3 P(\tau)$"
        )

        ax.set_title(
            f"Signed delay spectrum — SPW {key[0]}"
        )

        ax.grid(
            True,
            which="both",
            alpha=0.25
        )

        ax.legend()
        ax.set_yscale("log")

        fig.tight_layout()

        output_file = (
            out_dir
            / f"delta2_tau_signed_spw_{key[0]}.png"
        )

        fig.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()
        plt.close(fig)

        print(f"Saved: {output_file}")

    psc_delay._close()
    psc_eor._close()

# Plots delta^2(tau)

def plot_delta2_tau(
    unfiltered_pspec_file,
    delay_pspec_file,
    eor_pspec_file,
    out_dir,
    delay_hw,
):
    """
    Plot qualitative
        Delta^2(tau) ~ tau^3 P(tau)

    comparing delay-filtered EOR + foregrounds against
    unfiltered EOR only.

    delay_hw is in nanoseconds.
    """

    psc_unfiltered_sum = hp.PSpecContainer(unfiltered_pspec_file, mode='r')
    psc_delay = hp.PSpecContainer(delay_pspec_file, mode="r")
    psc_eor = hp.PSpecContainer(eor_pspec_file, mode="r")

    out_dir.mkdir(parents=True, exist_ok=True)

    uvp_unfiltered_sum = psc_unfiltered_sum.get_pspec(
        "stokespol",
        "time_and_interleave_averaged"
    )

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

        P_unfiltered = np.squeeze(
            uvp_unfiltered_sum.get_data(key).real
        )

        P_fg = np.squeeze(
            uvp_delay.get_data(key).real
        )

        P_eor = np.squeeze(
            uvp_eor.get_data(key).real
        )

        PN = np.squeeze(uvp_delay.get_stats("P_N", key))

        # ------------------------------------------------------------
        # Delay axis
        # ------------------------------------------------------------

        delays = np.squeeze(
            uvp_delay.get_dlys(key[0])
        )

        # Convert seconds -> ns
        delays_ns = delays * 1e9

        # ------------------------------------------------------------
        # Split negative and positive delays
        # ------------------------------------------------------------

        neg = delays_ns < 0
        pos = delays_ns > 0

        # Use |tau| as the x-axis for both branches
        tau_neg = np.abs(delays_ns[neg])
        tau_pos = delays_ns[pos]

        # ------------------------------------------------------------
        # Delta^2(tau) ~ |tau|^3 P(tau)
        # ------------------------------------------------------------

        delta2_fg_neg = tau_neg**3 * P_fg[neg]
        delta2_fg_pos = tau_pos**3 * P_fg[pos]

        PN_neg = tau_neg**3 * PN[neg]
        PN_pos = tau_pos**3 * PN[pos]

        delta2_eor_neg = tau_neg**3 * P_eor[neg]
        delta2_eor_pos = tau_pos**3 * P_eor[pos]

        delta2_unfiltered_neg = tau_neg**3 * P_unfiltered[neg]
        delta2_unfiltered_pos = tau_pos**3 * P_unfiltered[pos]

        min_len = min(
            len(tau_neg),
            len(tau_pos),
            len(delta2_fg_neg),
            len(delta2_fg_pos),
            len(delta2_eor_neg),
            len(delta2_eor_pos),
            len(PN_neg),
            len(PN_pos),
            len(delta2_unfiltered_neg),
            len(delta2_unfiltered_pos)
        )

        tau_neg = tau_neg[:min_len]
        tau_pos = tau_pos[:min_len]

        delta2_fg_neg = delta2_fg_neg[:min_len]
        delta2_fg_pos = delta2_fg_pos[:min_len]

        PN_neg = PN_neg[:min_len]
        PN_pos = PN_pos[:min_len]

        delta2_eor_neg = delta2_eor_neg[:min_len]
        delta2_eor_pos = delta2_eor_pos[:min_len]

        delta2_unfiltered_neg = delta2_unfiltered_neg[:min_len]
        delta2_unfiltered_pos = delta2_unfiltered_pos[:min_len]

        delta2_fg_avg = (delta2_fg_pos + delta2_fg_neg) / 2.0
        PN_avg = (PN_neg + PN_pos) / 2.0
        delta2_eor_avg = (delta2_eor_pos + delta2_eor_neg) / 2.0
        delta2_unfiltered_avg = (delta2_unfiltered_pos + delta2_unfiltered_neg) / 2.0

        delta2_neg_indices = delta2_fg_avg < 0
        tau_neg_pspec = tau_pos[delta2_neg_indices]
        delta2_fg_avg_neg = delta2_fg_avg[delta2_neg_indices]

        # ------------------------------------------------------------
        # Plot
        #
        # plot_positive_segments() handles NaN, Inf, zero and
        # negative values without deleting neighbouring points.
        # ------------------------------------------------------------

        fig, ax = plt.subplots(figsize=(9, 6))

        # ------------------------------------------------------------
        # Negative + positive averaged-delay branch
        # ------------------------------------------------------------

        ax.plot(
            tau_pos,
            abs(delta2_fg_avg),
            lw = 2,
            color = "red",
            label = rf"$abs(\Delta^2(\tau))$ for EOR + foregrounds (folded)"
        )

        ax.plot(
            tau_pos,
            abs(delta2_unfiltered_avg),
            lw = 2,
            linestyle = "--",
            color = "orange",
            label = rf"$abs(\Delta^2(\tau))$ for unfiltered EOR + foregrounds (folded)"
        )

        ax.plot(
            tau_pos,
            delta2_eor_avg,
            lw = 2,
            linestyle = "--",
            color = "red",
            label = rf"$\Delta^2(\tau)$ for EOR only (folded)"
        )

        ax.plot(
            tau_pos,
            PN_avg,
            linestyle = ":",
            lw = 1,
            color = "k",
            label = rf"$\Delta^2(\tau)$ for Noise ($P_N$) (folded)"
        )

        ax.scatter(
            tau_neg_pspec,
            abs(delta2_fg_avg_neg),
            lw = 2,
            color = "blue",
            label = rf"$\Delta^2(\tau) < 0$ for EOR + foregrounds (folded)"
        )

        ax.set_yscale('log')
        ax.set_xscale('log')

        # ------------------------------------------------------------
        # Delay-filter half width
        # ------------------------------------------------------------

        ax.axvline(
            delay_hw,
            color="k",
            linestyle="--",
            linewidth=1.5,
            label=fr"$\tau_\mathrm{{hw}}={delay_hw:.1f}$ ns"
        )

        # ------------------------------------------------------------
        # Labels
        # ------------------------------------------------------------

        ax.set_xlabel(r"$log(|\tau|)$ [ns]")

        ax.set_ylabel(
            r"$log(\Delta^2(\tau) \propto |\tau|^3 P(\tau))$"
        )

        ax.set_title(
            rf"Delay spectrum $\Delta^2(\tau)$ — SPW {key[0]}"
        )

        ax.grid(
            True,
            which="both",
            alpha=0.25
        )

        ax.legend()

        fig.tight_layout()

        output_file = (
            out_dir /
            f"delta2_tau_spw_{key[0]}.png"
        )

        fig.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()
        plt.close(fig)

        print(f"Saved: {output_file}")

    psc_delay._close()
    psc_eor._close()

def find_true_hw(uvp_sum, key):

    delays = uvp_sum.get_dlys(key[0]) * 1e9 #in ns

    #EOR + FG
    P_sum_real = abs(
        np.squeeze(
            uvp_sum.get_data(key).real
        )
    )

    P_sum_imag = abs(
        np.squeeze(
            uvp_sum.get_data(key).imag
        )
    )

    PN = np.squeeze(uvp_sum.get_stats("P_N", key))

    delays_positive = delays > 0
    delays_negative = delays < 0

    tau_neg = delays[delays_negative]
    tau_pos = delays[delays_positive]

    PN_neg = PN[delays_negative]
    PN_pos = PN[delays_positive]

    delta2_PN_neg = abs(abs(tau_neg)**3 * PN_neg)
    delta2_PN_pos = abs(tau_pos**3 * PN_pos)

    P_sum_real_neg = P_sum_real[delays_negative]
    P_sum_real_pos = P_sum_real[delays_positive]

    delta2_sum_real_neg = abs(abs(tau_neg)**3 * P_sum_real_neg)
    delta2_sum_real_pos = abs(tau_pos**3 * P_sum_real_pos)

    h_real_neg = abs(delta2_sum_real_neg - delta2_PN_neg)
    h_real_pos = abs(delta2_sum_real_pos - delta2_PN_pos)

    bin_real_neg_hw = np.argmin(h_real_neg)
    bin_real_pos_hw = np.argmin(h_real_pos)

    real_hw_avg = abs(abs(delays[delays_negative][bin_real_neg_hw]) + delays[delays_positive][bin_real_pos_hw]) / 2.0

    P_sum_imag_neg = P_sum_imag[delays_negative]
    P_sum_imag_pos = P_sum_imag[delays_positive]
    delta2_sum_imag_neg = abs(abs(tau_neg)**3 * P_sum_imag_neg)
    delta2_sum_imag_pos = abs(tau_pos**3 * P_sum_imag_pos)

    h_imag_neg = abs(delta2_sum_imag_neg - delta2_PN_neg)
    h_imag_pos = abs(delta2_sum_imag_pos - delta2_PN_pos)

    bin_imag_neg_hw = np.argmin(h_imag_neg)
    bin_imag_pos_hw = np.argmin(h_imag_pos)

    imag_hw_avg = abs(abs(delays[delays_negative][bin_imag_neg_hw]) + delays[delays_positive][bin_imag_pos_hw]) / 2.0

    return (imag_hw_avg + real_hw_avg) / 2.0
    


def signal_loss_test(
    delay_filtered_sum_pspec, 
    unfiltered_eor_pspec,
    hw_ns):
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

        true_hw = find_true_hw(uvp_sum, key)
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
        on_delay_file_pspec = bl_pair_folder_delay_on / f"{input_file.stem}_cutoff_{DLY_FILT_MIN_DLY}.tavg.pspec.h5"
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
                    str(bl_pair_folder_delay_off.absolute()),
                    str(DLY_FILT_MIN_DLY),
                    str(IS_INPAINT_HERE)
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
                    str(bl_pair_folder_delay_on.absolute()),
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

        plot_delta2_tau(
            off_delay_file_pspec,
            on_delay_file_pspec, 
            eor_file_pspec, 
            bl_pair_comparative_plots_folder, 
            hw
        )

        plot_delta2_tau_signed(
            off_delay_file_pspec,
            on_delay_file_pspec,
            eor_file_pspec,
            bl_pair_comparative_plots_folder,
            hw
        )

        signal_loss_test(
            on_delay_file_pspec, 
            off_delay_file_pspec, 
            hw
        )


def main():
    process_data()


if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("Usage: uv run signal_destruction_test.py\n")
        print("The program tests whether for any delay, the temperature is below")
        print("the boosted EOR signal using validation datasets.")
    main()