from hera_filters import dspec
import copy
import numpy as np
from scipy.signal.windows import dpss
import matplotlib.pyplot as plt

def frequency_to_delay(freqs, signal):
    #NOTE: AI GENERATED
    """
    Fourier transform a uniformly sampled frequency-space signal
    into delay space.

    Convention:
        V(nu) = integral d_tau Vtilde(tau) exp(+2 pi i nu tau)

    Returns:
        delays: delay coordinates in seconds
        delay_signal: complex signal in delay space
    """
    N = len(freqs)
    df = freqs[1] - freqs[0]

    # Shift frequency origin to zero before FFT
    signal_shifted = np.fft.ifftshift(signal)

    # Fourier transform frequency -> delay
    delay_signal = np.fft.fftshift(
        np.fft.ifft(signal_shifted)
    )

    # Account for the frequency-channel width
    delay_signal *= N * df

    # Delay coordinates
    delays = np.fft.fftshift(
        np.fft.fftfreq(N, d=df)
    )

    return delays, delay_signal


def delay_to_frequency(delays, delay_signal):
    """
    Inverse of frequency_to_delay: takes a delay-space signal defined on
    the same delay grid produced by frequency_to_delay (i.e.
    fftshift(fftfreq(N, d=df))) and returns the corresponding
    uniformly-sampled frequency-space signal.

    This exactly undoes the operations in frequency_to_delay, so
    round-tripping frequency_to_delay -> delay_to_frequency recovers the
    original signal (up to floating point error).
    """
    N = len(delays)
    dtau = delays[1] - delays[0]

    # Shift delay origin to zero before FFT
    delay_signal_shifted = np.fft.ifftshift(delay_signal)

    # Fourier transform delay -> frequency (inverse of the ifft used above)
    freq_signal = np.fft.fftshift(
        np.fft.fft(delay_signal_shifted)
    )

    # Account for the delay-bin width (inverse of the N*df scaling above)
    freq_signal *= dtau

    # Frequency coordinates
    freqs = np.fft.fftshift(
        np.fft.fftfreq(N, d=dtau)
    )

    return freqs, freq_signal


def delay_filter(freqs, signal_data, wgts, filter_centers, filter_half_widths, eigenval_cutoff, cache={}, zeros_where_zero_wgt=True):
    '''This function performs a high-pass delay filter, removing the wedge plus some buffer. It also performs inpainting with the same delay.'''
    dly_filt_data = copy.deepcopy(signal_data)
    inpainted_data = copy.deepcopy(signal_data)

    d_mdl = np.zeros_like(dly_filt_data)

    d_mdl, _, info = dspec.fourier_filter(freqs, signal_data, wgts=wgts, filter_centers=filter_centers,
                                                    filter_half_widths=filter_half_widths, mode='dpss_solve',
                                                    eigenval_cutoff=[eigenval_cutoff], suppression_factors=[eigenval_cutoff],
                                                    max_contiguous_edge_flags=len(freqs), cache=cache)
    if zeros_where_zero_wgt:
        dly_filt_data = np.where(wgts == 0, 0, dly_filt_data - d_mdl)
    else:
        dly_filt_data = dly_filt_data - d_mdl
    inpainted_data = np.where(wgts == 0, d_mdl, signal_data)
    #d_mdl is modeled. d_mdl is an array over delays. Thus, subtracting d_mdl from dly_filt_data is like subtracting actual data in delay space from the delay model.

    return dly_filt_data, inpainted_data


def create_dpss_signal_distribution(N, min_freq, max_freq, t_max, max_modes):
    '''Creates a distribution in frequency space around zero using DPSS functions'''
    dv = (max_freq - min_freq) / (N - 1) #Channel width
    NW = N * dv * t_max

    V = dpss(N, NW, Kmax = max_modes, sym = True)

    freqs = np.linspace(min_freq, max_freq, N)

    # Random DPSS coefficients
    c = (
    np.random.normal(size = max_modes) + 1j * np.random.normal(size = max_modes)) / np.sqrt(2)

    # Random signal in frequency space
    signal = c @ V

    #Plot signal
    fig, ax = plt.subplots()

    ax.plot(freqs, signal.real, color = 'red')
    plt.xlabel(rf"Frequency $\nu$")
    plt.ylabel(rf"DPSS generated signal")
    plt.title(rf"DPSS generated signal using max_modes number of random coefficients drawn from a normal distribution. $\tau = {t_max * 1e9}$.")
    plt.savefig('/home/Kwuzard/Projects/HERA_HONS/dpss_fluctuations_test_output/dpss_generated_signal.png', dpi=300, bbox_inches = 'tight')
    plt.show()
    plt.close(fig)
    return signal


def create_gaussian_signal_distribution(N, min_freq, max_freq, gaussian_half_width, delay_center=0.0):
    '''
    Creates a signal by building a Gaussian profile in delay space
    (centered at delay_center, with standard deviation gaussian_half_width)
    and Fourier transforming it into frequency space.

    Returns the frequency-space signal, sampled on the same freqs grid
    used elsewhere in this script.
    '''
    freqs = np.linspace(min_freq, max_freq, N)
    df = freqs[1] - freqs[0]

    # Delay grid consistent with frequency_to_delay's convention
    delays = np.fft.fftshift(np.fft.fftfreq(N, d=df))

    # Gaussian profile in delay space
    gaussian_delay_signal = np.exp(
        -0.5 * ((delays - delay_center) / gaussian_half_width) ** 2
    ).astype(complex)

    # Transform into frequency space to get the input signal
    freqs_check, signal = delay_to_frequency(delays, gaussian_delay_signal)
    # freqs_check should match freqs (same grid); use freqs for plotting/consistency

    #Plot signal
    fig, ax = plt.subplots()

    ax.plot(freqs, signal.real, color = 'red')
    plt.xlabel(rf"Frequency $\nu$")
    plt.ylabel(rf"Gaussian-profile generated signal")
    plt.title(
        rf"Signal generated from a Gaussian delay-space profile with "
        rf"width $\sigma = {gaussian_half_width * 1e9}$ ns."
    )
    plt.savefig('/home/Kwuzard/Projects/HERA_HONS/dpss_fluctuations_test_output/gaussian_generated_signal.png', dpi=300, bbox_inches = 'tight')
    plt.show()
    plt.close(fig)
    return signal


def main():
    #DPSS signal properties

    N = 512
    min_freq = 100e6
    max_freq = 200e6
    t_max = 500e-9
    k = 20

    # -----------------------------------------------------------------
    # Choose which input signal to use:
    #   'dpss'     -> the original random DPSS-generated signal, filtered
    #                 with the fixed half-width `hw` below.
    #   'gaussian' -> an input signal built from a Gaussian profile in
    #                 delay space, filtered using that same Gaussian's
    #                 width as the filter half-width.
    # -----------------------------------------------------------------
    signal_type = 'gaussian'  # 'dpss' or 'gaussian'

    # Width (seconds) used to build the Gaussian profile when
    # signal_type == 'gaussian'. This value is also used as the
    # delay-filter half-width so the filter matches the profile's scale.
    gaussian_half_width = 100e-9

    freqs = np.linspace(min_freq, max_freq, N)

    if signal_type == 'dpss':
        input_signal = create_dpss_signal_distribution(
            N,
            min_freq = min_freq,
            max_freq = max_freq,
            t_max = t_max,
            max_modes = k
        )
        hw = 100e-9  # fixed filter half-width, independent of t_max
    elif signal_type == 'gaussian':
        input_signal = create_gaussian_signal_distribution(
            N,
            min_freq = min_freq,
            max_freq = max_freq,
            gaussian_half_width = gaussian_half_width
        )
        hw = gaussian_half_width  # filter half-width matches the Gaussian's width
    else:
        raise ValueError(f"Unknown signal_type: {signal_type!r}. Use 'dpss' or 'gaussian'.")

    #Filter properties

    filter_centers = [0]
    filter_half_widths = [hw]
    wgts = np.ones(N)
    eigenvalue_cutoff = 1e-12
    zeros_where_zero_wgt = False

    filtered_signal = delay_filter(freqs, input_signal, wgts, filter_centers, filter_half_widths, eigenvalue_cutoff, zeros_where_zero_wgt = zeros_where_zero_wgt)

    #Go to delay space
    delays, input_delay = frequency_to_delay(freqs, input_signal)

    delays, filtered_delay = frequency_to_delay(
        freqs,
        filtered_signal[0]
    )

    #Plot filtered fig seperately in frequency space
    fig_filt, ax_filt = plt.subplots()
    ax_filt.plot(freqs, filtered_signal[0], color = 'red')
    plt.xlabel(rf"Frequency $\nu$")
    plt.ylabel(rf"Filtered signal")
    plt.title(f"{signal_type} generated signal now filtered using a DPSS filter of half-width {1e9 * filter_half_widths[0]} ns.")
    plt.savefig('/home/Kwuzard/Projects/HERA_HONS/dpss_fluctuations_test_output/dpss_filtered_signal.png', dpi=300, bbox_inches = 'tight')
    plt.show()
    plt.close(fig_filt)

    #Plots both filtered, unfiltered in delay space
    fig, ax = plt.subplots()

    ax.plot(
        delays * 1e9,
        np.abs(input_delay),
        label=f"Original {signal_type} signal"
    )

    ax.plot(
        delays * 1e9,
        np.abs(filtered_delay),
        linestyle = ':',
        label="After delay filter"
    )

    ax.axvline(hw * 1e9, linestyle='--', label=rf'$\tau=+{hw * 1e9}$ ns')
    ax.axvline(-hw * 1e9, linestyle='--', label=rf'$\tau=-{hw * 1e9}$ ns')

    ax.set_xlabel(r"Delay $\tau$ [ns]")
    ax.set_ylabel(r"$|\widetilde{V}(\tau)|$")
    ax.set_title("Delay-space effect of DPSS delay filter")
    ax.legend()

    #plt.ylim(0, 4 * 1e6)
    plt.xlim(-1000, 1000)
    plt.show()
    plt.savefig('/home/Kwuzard/Projects/HERA_HONS/dpss_fluctuations_test_output/dpss_test.png', dpi=300, bbox_inches = 'tight')
    plt.close(fig)

    diff = np.abs(filtered_delay) - np.abs(input_delay)

    # Don't plot the region |tau| < halfwidth
    mask = np.abs(delays) < hw
    diff[mask] = np.nan

    fig, ax = plt.subplots()

    ax.plot(
        delays * 1e9,
        diff,
        label="After delay filter"
    )

    ax.axvline(
        hw * 1e9,
        linestyle='--',
        label=rf'$\tau=+{hw * 1e9}$ ns'
    )

    ax.axvline(
        -hw * 1e9,
        linestyle='--',
        label=rf'$\tau=-{hw * 1e9}$ ns'
    )

    ax.set_xlabel(r"Delay $\tau$ [ns]")
    ax.set_ylabel(r"$|\Delta\widetilde{V}(\tau)|$")
    ax.set_title(
        "Difference (pre-post filter) in signal (delay space), "
        "ignoring the filtered region."
    )

    ax.set_xlim(-1000, 1000)
    ax.legend()

    plt.savefig(
        '/home/Kwuzard/Projects/HERA_HONS/dpss_fluctuations_test_output/dpss_test_diff.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()
    plt.close(fig)



if (__name__ == "__main__") :
    main()
