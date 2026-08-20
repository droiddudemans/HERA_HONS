from pathlib import Path
import sys
import subprocess
import hera_pspec as hp
import numpy as np
import os
import toml
import matplotlib.pyplot as plt
import pandas as pd

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

    print(uvp_yes.stats_array.keys())

    for key in uvp_yes.get_all_keys():
        print(key)
        print(uvp_yes.get_stats("P_N", key).shape)
        break
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

def power_ratio_calculation(no_delay_file, delay_file, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    if not no_delay_file.exists():
        raise FileNotFoundError(no_delay_file)

    if not delay_file.exists():
        raise FileNotFoundError(delay_file)

    output_power_ratio(no_delay_file, delay_file, out_dir)

def test_delay_cutoff():
    MAX_DELAY = 1000
    MIN_DELAY = 200
    DECR_DELAY = 100
    IS_INPAINT_HERE = False
    current_delay = MAX_DELAY

    #Loop over delays
    while current_delay >= MIN_DELAY:
        #Loop over baselines
        print(f"Running over all baselines with delay max cutoff : {current_delay}")

        for input_file in input_dir_raw.glob("*.uvh5"):
            delay_filter = False
            pair = input_file.name.split(".")[3]
            bl_pair_folder_name = "BLPAIR_" + pair
            ant1, ant2 = pair.split("_")

            out_dir = (
                Path(__file__).parent.parent.parent
                / "output"
                / "delay_cutoff_variation"
                / bl_pair_folder_name
                / f"cutoff_{current_delay}_ns"
            )

            if ant1 == ant2:
                continue  # Skip autocorrelations

            print(f"Processing file without delay filter : {input_file.name}")
            try:
                subprocess.run([
                    sys.executable,
                    "single_baseline_postprocessing_and_pspec.py",
                    str(input_file),
                    str(delay_filter),
                    str(output_dir_no_delay_filter.absolute()),
                    str(current_delay),
                    str(IS_INPAINT_HERE)
                ], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Exception in subproccess. Failed with error: {e.returncode}")
    
            delay_filter = True
            print(f"Processing file with delay filter : {input_file.name}")

            try:
                subprocess.run([
                    sys.executable,
                    "single_baseline_postprocessing_and_pspec.py",
                    str(input_file),
                    str(delay_filter),
                    str(output_dir_delay_filter.absolute()),
                    str(current_delay),
                    str(IS_INPAINT_HERE)
                ], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Exception in subproccess. Failed with error: {e.returncode}")


            no_delay = output_dir_no_delay_filter / f"{input_file.stem}.tavg.pspec.h5"
            delay = output_dir_delay_filter / f"{input_file.stem}.tavg.pspec.h5"

            power_ratio_calculation(no_delay, delay, out_dir)
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

    power_ratio_calculation(
        no_delay_file_name,
        delay_file_name,
        output_dir_inpaint_test,
    )

def z_analysis(no_delay_pspec_file, delay_pspec_file, z_out_dir, delay_hw):
    psc_yes = hp.PSpecContainer(delay_pspec_file, mode="r")
    psc_no = hp.PSpecContainer(no_delay_pspec_file, mode='r')

    z_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"The nominal half width was: {delay_hw} ns")

    uvp_yes = psc_yes.get_pspec(
        "stokespol",
        "time_and_interleave_averaged"
    )

    uvp_no = psc_no.get_pspec(
        "stokespol",
        "time_and_interleave_averaged"
    )

    # Create one combined normalized-difference figure.
    fig_norm_diff, ax_norm = plt.subplots(figsize=(10, 4))

    summary = []

    delay_hw_s = delay_hw * 1e-9

    colors = plt.cm.tab20(np.linspace(0, 1, len(uvp_no.get_all_keys()))) #Setting up colors for legends

    for i, key in enumerate(uvp_no.get_all_keys()):
        P_before = np.squeeze(uvp_no.get_data(key).real)
        P_after = np.squeeze(uvp_yes.get_data(key).real)

        PN_before = np.squeeze(uvp_no.get_stats("P_N", key))

        delays = uvp_no.get_dlys(key[0])

        delta = P_before - P_after
        sigma = PN_before
        z = delta / sigma

        # Work only with finite values when finding the effective cutoff.
        finite = np.isfinite(z) & np.isfinite(delays)
        delays_finite = delays[finite]
        z_finite = z[finite]

        #We still wanna plot points with huge pspec vals
        delays_plot_ns = delays_finite * 1e9
        z_plot = z_finite

        # Find the first delay on each side of zero where z is no longer
        # above +1.  The effective/true cutoff is the average of the
        # absolute delay on the two sides.
        #
        # Sort by distance from zero so that "first" means closest to the
        # centre of the delay spectrum.
        left = delays_finite < 0
        right = delays_finite > 0

        left_candidates = np.where(left & (z_finite <= 1))[0]
        right_candidates = np.where(right & (z_finite <= 1))[0]

        first_left_ns = np.nan
        first_right_ns = np.nan

        if len(left_candidates):
            left_idx = left_candidates[np.argmin(np.abs(delays_finite[left_candidates]))]
            first_left_ns = delays_finite[left_idx] * 1e9

        if len(right_candidates):
            right_idx = right_candidates[np.argmin(np.abs(delays_finite[right_candidates]))]
            first_right_ns = delays_finite[right_idx] * 1e9

        if np.isfinite(first_left_ns) and np.isfinite(first_right_ns):
            true_delay_cutoff_ns = (
                abs(first_left_ns) + abs(first_right_ns)
            ) / 2.0
            true_delay_cutoff_s = true_delay_cutoff_ns * 1e-9
        else:
            # If one side has no z <= 1 point, keep the configured cutoff
            # rather than inventing an asymmetric value.
            true_delay_cutoff_ns = delay_hw
            true_delay_cutoff_s = delay_hw_s

        # Keep only points outside the effective cutoff AND with z <= 1.
        # This removes the central region where the z signal is still above
        # the threshold, while preserving the fluctuations farther out.
        mask = (
            (np.abs(delays_finite) > true_delay_cutoff_s)
            & (np.abs(z_finite) <= 1)
        )

        delays_out = delays_finite[mask]
        z_out = z_finite[mask]

        if len(z_out) == 0:
            print(f"z_out was empty for spectral window {key[0]}")
            continue

        summary.append({
            "spw": key[0],
            "nominal_delay_hw_ns": delay_hw,
            "first_non_above_1_left_ns": first_left_ns,
            "first_non_above_1_right_ns": first_right_ns,
            "true_delay_cutoff_ns": true_delay_cutoff_ns,
            "N": len(z_out),
            "mean": np.mean(z_out),
            "median": np.median(z_out),
            "rms": np.std(z_out),
            "mad": np.median(np.abs(z_out - np.median(z_out))),
            "min": np.min(z_out),
            "max": np.max(z_out),
        })

        # ------------------------------------------------------------
        # Plot ALL points, including the filtered/central region.
        #
        # We clip only the plotted z values so large z values do not
        # dominate the y-axis. The original z values are unchanged and
        # are still used for the statistics above.
        # ------------------------------------------------------------

        delays_plot_ns = delays_finite * 1e9

        negative = delays_plot_ns < 0
        positive = delays_plot_ns > 0

        ax_norm.plot(
            delays_plot_ns[negative],
            z_plot[negative],
            lw=1.5,
            color=colors[i],
            label=f"SPW {key[0]}"
        )

        ax_norm.plot(
            delays_plot_ns[positive],
            z_plot[positive],
            lw=1.5,
            color=colors[i],
        )

        # Histogram (still one per spectral window).
        fig_histo, ax = plt.subplots(figsize=(5, 4))

        ax.hist(z_out, bins=30)
        ax.axvline(0, color='k', ls='--')

        ax.set_xlabel("z")
        ax.set_ylabel("Count")
        ax.set_title(f"Normalized Differences for SPW {key[0]}")

        fig_histo.tight_layout()

        fig_histo.savefig(
            z_out_dir / f"histo_normalized_diffs_spw_{key[0]}.png",
            dpi=300,
        )

        plt.close(fig_histo)

        # Save z values for this SPW.
        pd.DataFrame({
            "delay_s": delays_out,
            "z": z_out,
        }).to_csv(
            z_out_dir / f"z_values_spw_{key[0]}.csv",
            index=False,
        )
        
    ax_norm.set_ylim(-10, 10)
    # Finish the combined plot.
    ax_norm.axhline(0, color='k', ls='--')
    ax_norm.axhline(1, color='gray', ls=':')
    ax_norm.axhline(-1, color='gray', ls=':')

    # Keep the configured cutoff visible for comparison with the effective
    # cutoff used to select the plotted points.
    ax_norm.axvline(
        delay_hw,
        color='r',
        ls='--',
        alpha=0.6,
        label=f"Nominal cutoff = {delay_hw:.0f} ns"
    )
    ax_norm.axvline(
        -delay_hw,
        color='r',
        ls='--',
        alpha=0.6
    )

    ax_norm.set_xlabel("Delay (ns)")
    ax_norm.set_ylabel("z")
    ax_norm.set_title(
        f"Normalized Differences outside true cutoff, with z <= 1"
    )
    ax_norm.legend(loc="lower right", fontsize=8)

    fig_norm_diff.tight_layout()

    fig_norm_diff.savefig(
        z_out_dir / "norm_diff_all_spws.png",
        dpi=300,
    )

    plt.close(fig_norm_diff)

    # Save summary.
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(
        z_out_dir / "normalized_difference_summary.csv",
        index=False,
    )

    print(f"Saved summary to {z_out_dir/'normalized_difference_summary.csv'}")

def test_P_N():
    IS_INPAINT_HERE = False
    for input_file in input_dir_raw.glob("*.uvh5"):
        pair = input_file.name.split(".")[3]
        bl_pair_folder_name = "BLPAIR_" + pair
        ant1, ant2 = pair.split("_")

        z_out_dir = (
            Path(__file__).parent.parent.parent
            / "output"
            / "thermal_noise_test"
            / "z_out"
            / bl_pair_folder_name
        )

        output_dir_no_delay_filter = (
            Path(__file__).parent.parent.parent
            / "output"
            / "thermal_noise_test"
            / "non_delayed_pspec_data"
        )

        output_dir_delay_filter = (
            Path(__file__).parent.parent.parent
            / "output"
            / "thermal_noise_test"
            / "delayed_pspec_data"
        )

        output_dir_no_delay_filter.mkdir(parents=True, exist_ok=True)
        output_dir_delay_filter.mkdir(parents=True, exist_ok=True)

        if ant1 == ant2:
            continue  # Skip autocorrelations

        no_delay = output_dir_no_delay_filter / f"{input_file.stem}.tavg.pspec.h5"

        if not no_delay.is_file():
            print(f"Processing file without delay filter : {input_file.name}")
            try:
                subprocess.run([
                    sys.executable,
                    "single_baseline_postprocessing_and_pspec.py",
                    str(input_file),
                    "False",
                    str(output_dir_no_delay_filter.absolute()),
                    str(DLY_FILT_MIN_DLY),
                    str(IS_INPAINT_HERE)
                ], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Exception in subprocess. Failed with error: {e.returncode}")
        else:
            print(f"Skipping existing no-delay pspec: {no_delay.name}")
        
        delay = output_dir_delay_filter / f"{input_file.stem}_cutoff_{DLY_FILT_MIN_DLY}.tavg.pspec.h5"

        if not delay.is_file():
            print(f"Processing file with delay filter : {input_file.name}")
            try:
                subprocess.run([
                    sys.executable,
                    "single_baseline_postprocessing_and_pspec.py",
                    str(input_file),
                    "True",
                    str(output_dir_delay_filter.absolute()),
                    str(DLY_FILT_MIN_DLY),
                    str(IS_INPAINT_HERE)
                ], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Exception in subprocess. Failed with error: {e.returncode}")
        else:
            print(f"Skipping existing delay-filtered pspec: {delay.name}")

        no_delay = output_dir_no_delay_filter / f"{input_file.stem}.tavg.pspec.h5"
        delay = output_dir_delay_filter / f"{input_file.stem}_cutoff_{DLY_FILT_MIN_DLY}.tavg.pspec.h5"
        delay_hw = output_dir_delay_filter / f"{input_file.stem}_cutoff_{DLY_FILT_MIN_DLY}.tavg.delay_filter_hw.csv"

        hw = 0
        print("Got to before z analysis")
        with open(delay_hw, "r") as file:
            line = file.readline()
            hw = float(line.split(',')[1]) #Seond argument in csv file of first line is the half width of the delay filter.

        z_analysis(no_delay, delay, z_out_dir, hw)


def main(mode):
    if mode == 0:
        #Test if different minimum delay cutoffs have an effect on the P(after) / P(Before) for delay filtering.
        test_delay_cutoff()
    elif mode == 1:
        #Test whether inpainting decreases fluctuations on the P(After) / P(Before) for delay filtering.
        test_inpainting()
    elif mode == 2:
        test_P_N()

if __name__ == "__main__":
    if len(sys.argv) == 2: main(int(sys.argv[1]))