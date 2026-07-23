import numpy as np


def parse_band_str(
    band_str: str,
    freqs: np.ndarray,
) -> tuple[float, list[tuple[float, float]], list[float], list[float], list[slice], list[int]]:
    """Parse a band string into a channel width and a list of slices.

    Parameters
    ----------
    band_str : str
        A string of comma-separated frequency bands in the form 'low~high'.
    freqs : np.ndarray
        An array of frequencies in Hz.

    Returns
    -------
    df
        The channel width in Hz.
    bands
        A list of the frequency bands as (low, high) in MHz.
    min_freqs
        A list of the minimum frequencies in each band in MHz.
    max_freqs
        A list of the maximum frequencies in each band in MHz.
    band_slices
        A list of slices corresponding to the bands.
    nchans
        A list of the number of channels in each band.
    """
    df = np.median(np.diff(freqs))

    bands = [
        tuple(float(edge) for edge in band.split("~"))
        for band in band_str.strip().split(",")
    ]

    min_freqs = [
        (np.min(freqs[freqs >= 1e6 * b[0]]) - df / 2) / 1e6
        for b in bands
    ]

    max_freqs = [
        (np.max(freqs[freqs <= 1e6 * b[1]]) + df / 2) / 1e6
        for b in bands
    ]

    min_chan = [
        np.where(freqs >= 1e6 * b[0])[0][0]
        for b in bands
    ]

    max_chan = [
        np.where(freqs <= 1e6 * b[1])[0][-1]
        for b in bands
    ]

    band_slices = [
        slice(minc, maxc + 1)
        for minc, maxc in zip(min_chan, max_chan)
    ]

    nchans = [
        maxc - minc + 1
        for minc, maxc in zip(min_chan, max_chan)
    ]

    return (
        df,
        bands,
        min_freqs,
        max_freqs,
        band_slices,
        nchans,
    )