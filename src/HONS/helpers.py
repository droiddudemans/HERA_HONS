import numpy as np
import hera_pspec as hp

#-------------------HELPER FUNCTIONS-------------------#

def get_band_widths(band_str):
    """
    Parse BAND_STR into a list of bandwidths (MHz), one per spectral window,
    ordered to match spectral window index (key[0]).
    """
    widths = []
    for band in band_str.split(","):
        lo, hi = band.split("~")
        widths.append(float(hi) - float(lo))
    return widths

def find_true_hw(uvp_sum, key, hw):

    delays = uvp_sum.get_dlys(key[0]) * 1e9  # ns

    P_sum_real = np.abs(np.squeeze(uvp_sum.get_data(key).real))
    P_sum_imag = np.abs(np.squeeze(uvp_sum.get_data(key).imag))

    PN = np.squeeze(uvp_sum.get_stats("P_N", key))

    def first_crossing(delays, P, PN, side, hw, persistence=2):

        if side == "positive":
            mask = delays > hw
        else:
            mask = delays < -hw

        tau = delays[mask]
        P = P[mask]
        PN_side = PN[mask]

        # Make sure we move outward in |tau|
        order = np.argsort(np.abs(tau))
        tau = tau[order]
        P = P[order]
        PN_side = PN_side[order]

        diff = P - PN_side

        # Look for the first genuine sign change
        for i in range(len(diff) - 1):

            if diff[i] * diff[i + 1] > 0:
                continue

            # Optional persistence check to reject tiny wiggles
            new_sign = np.sign(diff[i + 1])

            if new_sign == 0:
                return abs(tau[i + 1])

            end = min(i + 1 + persistence, len(diff))

            if np.all(np.sign(diff[i + 1:end]) == new_sign):

                # Linear interpolation between the two bins
                x1, x2 = tau[i], tau[i + 1]
                y1, y2 = diff[i], diff[i + 1]

                if y2 != y1:
                    crossing = x1 - y1 * (x2 - x1) / (y2 - y1)
                else:
                    crossing = 0.5 * (x1 + x2)

                return abs(crossing)

        return np.nan

    def symmetric_hw(P):

        neg = first_crossing(
            delays, P, PN,
            side="negative",
            hw=hw
        )

        pos = first_crossing(
            delays, P, PN,
            side="positive",
            hw=hw
        )

        if np.isfinite(neg) and np.isfinite(pos):
            return 0.5 * (neg + pos)

        elif np.isfinite(neg):
            return neg

        elif np.isfinite(pos):
            return pos

        return np.nan

    # Find symmetric first-intersection width
    real_hw = symmetric_hw(P_sum_real)
    imag_hw = symmetric_hw(P_sum_imag)

    values = [x for x in (real_hw, imag_hw) if np.isfinite(x)]

    if not values:
        return np.nan

    return np.mean(values)


#-----------------------------------------------------#
#------------POWER SPECTRUM FUNCTIONS-----------------#
#-----------------------------------------------------#


def get_power_lims_beyond_hw(power_arr, delays_ns, hw, delay_lim = 2.5):
    """
    Calculates the min, max power beyond hw * delay_lim in delay space. 
    Typically used for plotting and ignoring huge fg in plot range choice.
    Returns as [ylim_min, ylim_max]
    """
    ylim_max = 0
    ylim_min = 0
    for i in range(len(power_arr)):
        delay = delays_ns[i]
        if delay <= delay_lim * hw : continue
        power_at_delay = abs(power_arr[i])

        ylim_max = max([ylim_max, power_at_delay])
        ylim_min = min([ylim_min, power_at_delay])
    return ylim_min, ylim_max


def count_num_amplified_bl(no_delay_pspec_file, delay_pspec_file):
    '''Counts & returns how many bins have the filtered signal larger than the unfiltered accross all spectral windows for the given baseline.'''
    psc_yes = hp.PSpecContainer(delay_pspec_file, mode="r")
    psc_no = hp.PSpecContainer(no_delay_pspec_file, mode='r')

    uvp_yes = psc_yes.get_pspec(
        "stokespol",
        "time_and_interleave_averaged"
    )

    uvp_no = psc_no.get_pspec(
        "stokespol",
        "time_and_interleave_averaged"
    )

    count_larger = 0

    for _, key in enumerate(uvp_no.get_all_keys()):
        P_before = np.squeeze(uvp_no.get_data(key).real)
        P_after = np.squeeze(uvp_yes.get_data(key).real)

        larger = P_after > P_before
        P_larger = P_after[larger]
        count_larger += len(P_larger)

    psc_yes._close()
    psc_no._close()

    return count_larger


def calculate_leakage_scores(
    delay_filtered_sum_pspec,
    unfiltered_sum_pspec,
    hw_ns,
    weight_mode="count",
):
    """
    Calculate leakage scores for each spectral window.

    Parameters
    ----------
    delay_filtered_sum_pspec
        Filtered power-spectrum data.
    unfiltered_sum_pspec
        Unfiltered power-spectrum data.
    hw_ns : float
        Horizon wedge delay in ns.
    weight_mode : str
        How to score each bin where filtered > unfiltered:
        - "count" : each such bin contributes 1
        - "ratio" : each such bin contributes filtered / unfiltered
        - "diff"  : each such bin contributes filtered - unfiltered

    Returns
    -------
    spw_indices : list
        Spectral window indices.
    scores : list
        Leakage score corresponding to each spectral window.
    """
    if weight_mode not in ("count", "ratio", "diff"):
        raise ValueError(
            f"weight_mode must be 'count', 'ratio', or 'diff', got {weight_mode!r}"
        )

    psc_filtered = hp.PSpecContainer(delay_filtered_sum_pspec, mode="r")
    psc_unfiltered = hp.PSpecContainer(unfiltered_sum_pspec, mode="r")

    uvp_sum = psc_filtered.get_pspec(
        "stokespol",
        "time_and_interleave_averaged",
    )

    uvp_filtered_folded = psc_filtered.get_pspec(
        "stokespol",
        "time_and_interleave_averaged",
    )
    uvp_filtered_folded.fold_spectra()

    uvp_unfiltered_folded = psc_unfiltered.get_pspec(
        "stokespol",
        "time_and_interleave_averaged",
    )
    uvp_unfiltered_folded.fold_spectra()

    spw_indices = []
    scores = []

    for key in uvp_sum.get_all_keys():
        spw = key[0]

        P_sum_real = np.squeeze(
            uvp_filtered_folded.get_data(key).real
        )
        P_eor_real = np.squeeze(
            uvp_unfiltered_folded.get_data(key).real
        )

        delays = np.squeeze(
            uvp_filtered_folded.get_dlys(spw)
        )
        delays_ns = delays * 1e9

        delta_2_sum_real = abs(delays_ns**3 * P_sum_real)
        delta_2_eor_real = abs(delays_ns**3 * P_eor_real)

        true_hw = find_true_hw(uvp_sum, key, hw_ns)
        bin_hw_index = np.argmin(
            np.abs(delays_ns - true_hw)
        )

        score = 0.0

        for i in range(bin_hw_index, len(delays_ns)):
            filt = delta_2_sum_real[i]
            unfilt = delta_2_eor_real[i]

            if filt > unfilt:
                if weight_mode == "count":
                    score += 1

                elif weight_mode == "ratio":
                    # Guard against division by zero.
                    if unfilt != 0:
                        score += filt / unfilt

                elif weight_mode == "diff":
                    score += filt - unfilt

        print(f"Spectral window index: {spw}")
        print(f"Signal loss score ({weight_mode}): {score}\n")

        spw_indices.append(spw)
        scores.append(score)

    psc_filtered._close()
    psc_unfiltered._close()

    return spw_indices, scores

def signal_loss_test(
    delay_filtered_sum_pspec, 
    hw_ns,
    output_folder):
    psc_sum = hp.PSpecContainer(delay_filtered_sum_pspec, mode="r")

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
