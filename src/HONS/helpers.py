import numpy as np
import hera_pspec as hp

#-------------------HELPER FUNCTIONS-------------------#

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