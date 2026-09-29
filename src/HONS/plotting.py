import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import hera_pspec as hp
import helpers

# ======================================================================
# Color helper — guarantees no two curves ever share (or nearly share)
# a color, across an arbitrary number of filtered files.
# ======================================================================

def _get_distinct_colors(n_needed, reserved_colors=None, tol=0.15):
    """
    Return `n_needed` colors guaranteed to be mutually distinct and
    distinct from any colors in `reserved_colors`.

    Pulls from several qualitative colormaps (tab10/tab20/tab20b/tab20c)
    for a large pool of visually separable colors, rather than relying
    on matplotlib's default 10-color cycle (which silently repeats).

    Raises
    ------
    ValueError
        If more colors are requested than are available in the pool
        after excluding anything too close to a reserved color. This
        is intentional: silently repeating a color defeats the point.
    """
    reserved_colors = reserved_colors or []
    reserved_rgb = [mcolors.to_rgb(c) for c in reserved_colors]

    candidates = []
    for cmap_name in ("tab10", "tab20", "tab20b", "tab20c"):
        candidates.extend(plt.get_cmap(cmap_name).colors)

    def _too_close(c1, c2):
        return all(abs(a - b) < tol for a, b in zip(c1, c2))

    seen = set()
    pool = []
    for c in candidates:
        c = tuple(c)
        if c in seen:
            continue
        seen.add(c)
        if any(_too_close(c, r) for r in reserved_rgb):
            continue
        pool.append(c)

    if n_needed > len(pool):
        raise ValueError(
            f"Requested {n_needed} mutually distinct colors, but only "
            f"{len(pool)} are available in the palette once colors too "
            f"close to {reserved_colors} are excluded. Reduce the number "
            f"of curves being plotted together, or extend the palette "
            f"in _get_distinct_colors()."
        )

    return pool[:n_needed]


# Reserved colors used by every plot in this module so the
# "reference" curves are always visually consistent and never get
# reassigned to a filtered curve.
UNFILTERED_COLOR = "black"
EOR_COLOR = "darkorange"
BOUNDARY_COLOR = "dimgray"
ZERO_LINE_COLOR = "lightgray"

_RESERVED_COLORS = [UNFILTERED_COLOR, EOR_COLOR, BOUNDARY_COLOR, ZERO_LINE_COLOR]

def _fold_onto_abs_tau(delays_ns, delta2):
    """
    Fold a signed delay axis onto |tau|.

    Any delay that has both a positive- and negative-delay counterpart
    (a genuine +/- pair) is averaged. A delay that has no counterpart
    -- zero delay, or an unpaired Nyquist bin when the axis isn't
    perfectly symmetric -- is kept as-is instead of being dropped.
    This replaces position-based pairing + min_len truncation, which
    silently lost points whenever the negative/positive halves didn't
    have exactly matching lengths.

    Parameters
    ----------
    delays_ns : np.ndarray
        Signed delay axis, in ns.
    delta2 : np.ndarray
        Delta^2(tau) values (or any per-delay quantity) aligned with
        delays_ns.

    Returns
    -------
    tau : np.ndarray
        Sorted, unique |delay| values.
    folded : np.ndarray
        Corresponding folded/averaged values, same length as tau.
    """
    abs_tau = np.abs(delays_ns)
    unique_tau, inverse = np.unique(abs_tau, return_inverse=True)

    folded = np.zeros(unique_tau.shape, dtype=delta2.dtype)
    counts = np.zeros(unique_tau.shape, dtype=int)
    np.add.at(folded, inverse, delta2)
    np.add.at(counts, inverse, 1)
    folded = folded / counts

    return unique_tau, folded


def plot_signal_loss(unfiltered_pspec, filtered_pspecs, eigenvalue_cutoffs, delay_standoffs, filter_hw):

    psc_filtered = hp.PSpecContainer(filtered_pspec, mode = "r")
    psc_unfiltered = hp.PSpecContainer(unfiltered_pspec, mode = "r")

    uvp_filtered_folded = psc_filtered.get_pspec(
        "stokespol",
        "time_and_interleave_averaged"
    )

    uvp_unfiltered_fodled = psc_unfiltered.get_pspec(
        "stokespol",
        "time_and_interleave_averaged"
    )

    uvp_filtered_folded.fold_spectra()
    uvp_unfiltered_fodled.fold_spectra()

    for key in uvp_filtered_folded.get_all_keys():
        P_filtered_real = np.squeeze(
            uvp_filtered_folded.get_data(key).real
        )

        P_unfiltered_real = np.squeeze(
            uvp_unfiltered_fodled.get_data(key).real
        )

        P_filtered_imag = np.squeeze(
            uvp_filtered_folded.get_data(key).imag
        )

        P_unfiltered_imag = np.squeeze(
            uvp_unfiltered_fodled.get_data(key).imag
        )

        signal_loss_real = np.ones_like(P_filtered_real) - (P_filtered_real / P_unfiltered_real)
        signal_loss_imag = np.ones_like(P_filtered_imag) - (P_filtered_imag / P_unfiltered_imag)
        signal_loss_avg = (signal_loss_real + signal_loss_imag) / 2.0

        delays = np.squeeze(
            uvp_filtered_folded.get_dlys(key[0])
        )

        delays_ns = delays * 1e9
        

    psc_filtered._close()
    psc_unfiltered._close()


def plot_delta2_tau_multiple(
    unfiltered_pspec_file,
    delay_pspec_files,
    delay_hw,
    out_dir = None,
    labels = None,
    eor_pspec_file = None,
    unfiltered_label = "Unfiltered EOR + FG",
    eor_label = "Unfiltered EOR only",
    log_scale = False,
    scale_y_axis_detailed = False
):
    """
    Plot folded/unsigned Delta^2(tau) for multiple delay-filtered
    PSpec files together with the unfiltered signal.

    Every delay bin is plotted: matched +/- pairs are averaged, and
    any unmatched bin (zero delay, or a lone Nyquist bin) is kept
    rather than dropped.

    Parameters
    ----------
    unfiltered_pspec_file : Path
        PSpec file containing the unfiltered EOR + foreground signal.

    delay_pspec_files : list[Path]
        List of delay-filtered PSpec files to compare.

    delay_hw : float or list[float]
        Nominal delay-filter half-width in ns. Either a single value
        applied to every entry in delay_pspec_files, or a list with
        one half-width per entry (ordered to match delay_pspec_files).

    out_dir : Path, optional
        Directory where plots are saved.
    labels : list[str], optional
        Legend labels for the delay-filtered curves, one per entry in
        delay_pspec_files. If None, the filenames are used.

    unfiltered_label : str, optional
        Legend label for the unfiltered curve.

    eor_label : str, optional
        Legend label for the unfiltered-EOR-only curve.

    eor_pspec_file : Path, optional
        PSpec file containing the unfiltered EOR signal.

    log_scale : bool, optional
        If True, use log-log axes. Note the tau=0 point (if present)
        cannot be shown on a log x-axis and matplotlib will simply
        omit it in that case.

    scale_y_axis_detailed : bool, optional
        Sets the y-lims of the plots to be something of the limits you'd find
        beyond the fg wedge.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Normalize delay_hw into a per-file list
    # ------------------------------------------------------------
    delay_hw_is_scalar = np.isscalar(delay_hw)
    if delay_hw_is_scalar:
        delay_hws = [delay_hw] * len(delay_pspec_files)
    else:
        delay_hws = list(delay_hw)
        if len(delay_hws) != len(delay_pspec_files):
            raise ValueError(
                "delay_hw must be a single float or a list with the same "
                "length as delay_pspec_files."
            )

    psc_unfiltered = hp.PSpecContainer(unfiltered_pspec_file, mode="r")
    psc_eor = hp.PSpecContainer(eor_pspec_file, mode="r") if eor_pspec_file else None
    psc_delays = [
        hp.PSpecContainer(delay_pspec_file, mode="r")
        for delay_pspec_file in delay_pspec_files
    ]

    uvp_unfiltered = psc_unfiltered.get_pspec("stokespol", "time_and_interleave_averaged")
    uvp_eor = psc_eor.get_pspec("stokespol", "time_and_interleave_averaged") if eor_pspec_file else None
    uvp_delays = [
        psc.get_pspec("stokespol", "time_and_interleave_averaged")
        for psc in psc_delays
    ]

    if labels is None:
        labels = [Path(f).stem for f in delay_pspec_files]

    if len(labels) != len(delay_pspec_files):
        raise ValueError("Number of labels must match number of delay_pspec_files.")

    filtered_colors = _get_distinct_colors(
        len(delay_pspec_files), reserved_colors=_RESERVED_COLORS
    )

    if delay_hw_is_scalar:
        print(f"Nominal delay half-width: {delay_hw} ns")
    else:
        print(f"Nominal delay half-widths: {delay_hws} ns")

    for key in uvp_delays[0].get_all_keys():

        # ------------------------------------------------------------
        # Delay axis
        # ------------------------------------------------------------

        delays = np.squeeze(uvp_delays[0].get_dlys(key[0]))
        delays_ns = delays * 1e9

        # ------------------------------------------------------------
        # Unfiltered
        # ------------------------------------------------------------

        P_unfiltered = np.squeeze(uvp_unfiltered.get_data(key).real)
        delta2_unfiltered_raw = np.abs(delays_ns) ** 3 * P_unfiltered
        tau, delta2_unfiltered_avg = _fold_onto_abs_tau(delays_ns, delta2_unfiltered_raw)

        # ------------------------------------------------------------
        # Unfiltered EOR only
        # ------------------------------------------------------------

        P_eor = np.squeeze(uvp_eor.get_data(key).real) if eor_pspec_file else None
        delta2_eor_raw = np.abs(delays_ns) ** 3 * P_eor if eor_pspec_file else None

        if eor_pspec_file:
            _, delta2_eor_avg = _fold_onto_abs_tau(delays_ns, delta2_eor_raw)
        else:
            delta2_eor_avg = None

        # ------------------------------------------------------------
        # Set y-lims if desired
        # ------------------------------------------------------------
        DELAY_LIM = 2.5
        ylim_max = 0
        ylim_min = 0
        if scale_y_axis_detailed:

            #UNFILTERED POWER
            ylim_min, ylim_max = helpers.get_power_lims_beyond_hw(
                delta2_unfiltered_avg, 
                tau, 
                delay_hw, 
                delay_lim = DELAY_LIM
            )

            #EOR POWER
            if eor_pspec_file:
                ylim_min, ylim_max = helpers.get_power_lims_beyond_hw(
                    delta2_eor_avg, 
                    tau, 
                    delay_hw, 
                    delay_lim = DELAY_LIM
                )

        # ------------------------------------------------------------
        # Plot
        # ------------------------------------------------------------

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.plot(
            tau,
            np.abs(delta2_unfiltered_avg),
            lw=2.5,
            color=UNFILTERED_COLOR,
            label=unfiltered_label,
            zorder=3,
        )

        if eor_pspec_file:
            ax.plot(
                tau,
                np.abs(delta2_eor_avg),
                lw=2.6,
                color=EOR_COLOR,
                linestyle=":",
                label=eor_label,
                zorder=3,
            )

        # ------------------------------------------------------------
        # Filtered curves — each gets its own guaranteed-unique color
        # ------------------------------------------------------------

        for uvp_delay, label, color in zip(uvp_delays, labels, filtered_colors):

            P_filtered = np.squeeze(uvp_delay.get_data(key).real)
            delta2_filtered_raw = np.abs(delays_ns) ** 3 * P_filtered
            _, delta2_filtered_avg = _fold_onto_abs_tau(delays_ns, delta2_filtered_raw)

            if scale_y_axis_detailed:
                #FILTERED POWER
                ylim_min, ylim_max = helpers.get_power_lims_beyond_hw(
                    delta2_filtered_avg, 
                    tau, 
                    delay_hw, 
                    delay_lim = DELAY_LIM
                )

            ax.plot(
                tau,
                np.abs(delta2_filtered_avg),
                lw=2,
                color=color,
                label=label,
                zorder=2,
            )

        # ---------------- Set y-limits on plots ---------------------
        if scale_y_axis_detailed:
            ax.set_ylim(ylim_min, ylim_max)

        # ------------------------------------------------------------
        # Delay-filter boundary/boundaries
        # ------------------------------------------------------------

        if delay_hw_is_scalar:
            # Single shared half-width: one boundary line, as before.
            ax.axvline(
                delay_hws[0],
                color=BOUNDARY_COLOR,
                linestyle="--",
                linewidth=1.5,
                label=rf"$\tau_\mathrm{{hw}}={delay_hws[0]:.1f}$ ns",
                zorder=1,
            )
        else:
            # One boundary line per filtered curve, colored to match
            # its curve so it's clear which half-width belongs to
            # which filter.
            for hw, color in zip(delay_hws, filtered_colors):
                ax.axvline(
                    hw,
                    color=color,
                    linestyle="--",
                    linewidth=1.5,
                    label=rf"$\tau_\mathrm{{hw}}={hw:.1f}$ ns",
                    zorder=1,
                )

        # ------------------------------------------------------------
        # Axes
        # ------------------------------------------------------------

        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(r"$log(|\tau|)$ [ns]")
            ax.set_ylabel(r"$log(|\Delta^2(\tau)|) \propto log(|\tau|^3 P(\tau))$")
        else:
            ax.set_xlabel(r"|\tau|$ [ns]")
            ax.set_ylabel(r"$|\Delta^2(\tau)| \propto |\tau|^3 P(\tau)$")
        ax.set_title(rf"Delay spectrum $\Delta^2(\tau)$ — SPW {key[0]}")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()

        fig.tight_layout()

        if out_dir:
            output_file = out_dir / f"delta2_tau_multiple_spw_{key[0]}.png"
            fig.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"Saved: {output_file}")
            plt.close()
        else:
            plt.show()

    psc_unfiltered._close()
    if eor_pspec_file:
        psc_eor._close()
    for psc in psc_delays:
        psc._close()


def plot_delta2_tau_signed_multiple(
    unfiltered_pspec_file,
    delay_pspec_files,
    delay_hw,
    out_dir = None,
    eor_pspec_file = None,
    labels = None,
    unfiltered_label = "Unfiltered EOR + FG",
    eor_label = "Unfiltered EOR only",
    log_scale = False,
    scale_y_axis_detailed = True,
):
    """
    Plot Delta^2(tau) against the signed delay axis for multiple
    delay-filtered PSpec files, together with the unfiltered signal.

    The negative- and positive-delay halves each include the tau=0
    point (rather than strictly excluding it), so every delay bin is
    drawn and the two halves connect through the origin with no gap.

    Parameters
    ----------
    unfiltered_pspec_file : Path
        PSpec file containing the unfiltered EOR + foreground signal.

    delay_pspec_files : list[Path]
        List of delay-filtered PSpec files to compare.

    out_dir : Path, optional
        Directory where plots are saved.

    delay_hw : float or list[float]
        Nominal delay-filter half-width in ns. Either a single value
        applied to every entry in delay_pspec_files, or a list with
        one half-width per entry (ordered to match delay_pspec_files).

    labels : list[str], optional
        Legend labels for the delay-filtered curves, one per entry in
        delay_pspec_files. If None, the filenames are used.
    
    eor_pspec_file : Path, optional
        PSpec file containing the unfiltered EOR signal.

    unfiltered_label : str, optional
        Legend label for the unfiltered curve.

    eor_label : str, optional
        Legend label for the unfiltered-EOR-only curve.

    scale_y_axis_detailed : bool, optional
        Sets the y-lims of the plots to be something of the limits you'd find
        beyond the fg wedge.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Normalize delay_hw into a per-file list
    # ------------------------------------------------------------
    delay_hw_is_scalar = np.isscalar(delay_hw)
    if delay_hw_is_scalar:
        delay_hws = [delay_hw] * len(delay_pspec_files)
    else:
        delay_hws = list(delay_hw)
        if len(delay_hws) != len(delay_pspec_files):
            raise ValueError(
                "delay_hw must be a single float or a list with the same "
                "length as delay_pspec_files."
            )

    psc_unfiltered = hp.PSpecContainer(unfiltered_pspec_file, mode="r")
    psc_eor = hp.PSpecContainer(eor_pspec_file, mode="r") if eor_pspec_file else None
    psc_delays = [
        hp.PSpecContainer(delay_pspec_file, mode="r")
        for delay_pspec_file in delay_pspec_files
    ]

    uvp_unfiltered = psc_unfiltered.get_pspec("stokespol", "time_and_interleave_averaged")
    uvp_eor = psc_eor.get_pspec("stokespol", "time_and_interleave_averaged") if eor_pspec_file else None
    uvp_delays = [
        psc.get_pspec("stokespol", "time_and_interleave_averaged")
        for psc in psc_delays
    ]

    if labels is None:
        labels = [Path(f).stem for f in delay_pspec_files]

    if len(labels) != len(delay_pspec_files):
        raise ValueError("Number of labels must match number of delay_pspec_files.")

    filtered_colors = _get_distinct_colors(
        len(delay_pspec_files), reserved_colors=_RESERVED_COLORS
    )

    if delay_hw_is_scalar:
        print(f"Nominal delay half-width: {delay_hw} ns")
    else:
        print(f"Nominal delay half-widths: {delay_hws} ns")

    for key in uvp_delays[0].get_all_keys():

        # ------------------------------------------------------------
        # Delay axis
        # ------------------------------------------------------------

        delays = np.squeeze(uvp_delays[0].get_dlys(key[0]))
        delays_ns = delays * 1e9

        # ------------------------------------------------------------
        # Unfiltered signal
        # ------------------------------------------------------------

        P_unfiltered = np.squeeze(uvp_unfiltered.get_data(key).real)
        delta2_unfiltered = np.abs(delays_ns) ** 3 * P_unfiltered

        negative_unfiltered = np.isfinite(delta2_unfiltered) & (delays_ns <= 0)
        positive_unfiltered = np.isfinite(delta2_unfiltered) & (delays_ns >= 0)

        # ------------------------------------------------------------
        # Unfiltered EOR only signal
        # ------------------------------------------------------------

        P_eor = np.squeeze(uvp_eor.get_data(key).real) if eor_pspec_file else None
        delta2_eor = np.abs(delays_ns) ** 3 * P_eor if eor_pspec_file else None

        negative_eor = np.isfinite(delta2_eor) & (delays_ns <= 0) if eor_pspec_file else None
        positive_eor = np.isfinite(delta2_eor) & (delays_ns >= 0) if eor_pspec_file else None

        # ------------------------------------------------------------
        # Create plot-space
        # ------------------------------------------------------------

        fig, ax = plt.subplots(figsize=(10, 6))

        # ------------------------------------------------------------
        # Set y-lims if desired
        # ------------------------------------------------------------
        DELAY_LIM = 4

        ylim_max = 0
        ylim_min = 0
        if scale_y_axis_detailed:

            #UNFILTERED POWER
            #Checking negative axis
            ylim_min, ylim_max = helpers.get_power_lims_beyond_hw(
                delta2_unfiltered[negative_unfiltered], 
                delays_ns[negative_unfiltered], 
                delay_hw, 
                delay_lim = DELAY_LIM
            )

            #Checking positive axis
            ylim_min, ylim_max = helpers.get_power_lims_beyond_hw(
                delta2_unfiltered[positive_unfiltered], 
                delays_ns[positive_unfiltered], 
                delay_hw, 
                delay_lim = DELAY_LIM
            )

            #EOR POWER
            if eor_pspec_file:
                ylim_min, ylim_max = helpers.get_power_lims_beyond_hw(
                    delta2_eor[negative_eor], 
                    delays_ns[negative_eor], 
                    delay_hw, 
                    delay_lim = DELAY_LIM
                )

        # ------------------------------------------------------------
        # PLOT
        # ------------------------------------------------------------

        # Unfiltered reference
        ax.plot(
            delays_ns[negative_unfiltered],
            np.abs(delta2_unfiltered[negative_unfiltered]),
            lw=2.5,
            color=UNFILTERED_COLOR,
            label=unfiltered_label,
            zorder=3,
        )
        ax.plot(
            delays_ns[positive_unfiltered],
            np.abs(delta2_unfiltered[positive_unfiltered]),
            lw=2.5,
            color=UNFILTERED_COLOR,
            zorder=3,
        )

        # Unfiltered EOR only reference
        if eor_pspec_file:
            ax.plot(
                delays_ns[negative_eor],
                np.abs(delta2_eor[negative_eor]),
                lw=2.5,
                color=EOR_COLOR,
                linestyle=":",
                label=eor_label,
                zorder=3,
            )
            ax.plot(
                delays_ns[positive_eor],
                np.abs(delta2_eor[positive_eor]),
                lw=2.5,
                color=EOR_COLOR,
                linestyle=":",
                zorder=3,
            )

        # ------------------------------------------------------------
        # Filtered curves — each gets its own unique color,
        # shared between its negative- and positive-delay halves
        # ------------------------------------------------------------

        for uvp_delay, label, color in zip(uvp_delays, labels, filtered_colors):

            P_filtered = np.squeeze(uvp_delay.get_data(key).real)
            delta2_filtered = np.abs(delays_ns) ** 3 * P_filtered

            negative_filtered = np.isfinite(delta2_filtered) & (delays_ns <= 0)
            positive_filtered = np.isfinite(delta2_filtered) & (delays_ns >= 0)

            if scale_y_axis_detailed:
                #FILTERED POWER
                #Checking negative axis
                ylim_min, ylim_max = helpers.get_power_lims_beyond_hw(
                    delta2_filtered[negative_filtered], 
                    delays_ns[negative_filtered], 
                    delay_hw, 
                    delay_lim = DELAY_LIM
                )
    
                #Checking positive axis
                ylim_min, ylim_max = helpers.get_power_lims_beyond_hw(
                    delta2_filtered[positive_filtered], 
                    delays_ns[positive_filtered], 
                    delay_hw, 
                    delay_lim = DELAY_LIM
                )
    

            ax.plot(
                delays_ns[negative_filtered],
                np.abs(delta2_filtered[negative_filtered]),
                lw=2,
                color=color,
                label=label,
                zorder=2,
            )
            ax.plot(
                delays_ns[positive_filtered],
                np.abs(delta2_filtered[positive_filtered]),
                lw=2,
                color=color,
                zorder=2,
            )

        # ---------------- Set y-limits on plots ---------------------
        if scale_y_axis_detailed:
            ax.set_ylim(ylim_min, ylim_max)


        # ------------------------------------------------------------
        # Delay-filter boundary/boundaries
        # ------------------------------------------------------------

        if delay_hw_is_scalar:
            # Single shared half-width: one +/- pair, as before.
            ax.axvline(
                -delay_hws[0],
                color=BOUNDARY_COLOR,
                linestyle="--",
                linewidth=1.5,
                label=rf"$\pm\tau_\mathrm{{hw}}={delay_hws[0]:.1f}$ ns",
                zorder=1,
            )
            ax.axvline(
                delay_hws[0],
                color=BOUNDARY_COLOR,
                linestyle="--",
                linewidth=1.5,
                zorder=1,
            )
        else:
            # One +/- pair per filtered curve, colored to match its
            # curve so it's clear which half-width belongs to which
            # filter.
            for hw, color in zip(delay_hws, filtered_colors):
                ax.axvline(
                    -hw,
                    color=color,
                    linestyle="--",
                    linewidth=1.5,
                    label=rf"$\pm\tau_\mathrm{{hw}}={hw:.1f}$ ns",
                    zorder=1,
                )
                ax.axvline(
                    hw,
                    color=color,
                    linestyle="--",
                    linewidth=1.5,
                    zorder=1,
                )

        # Zero-delay reference
        ax.axvline(
            0,
            color=ZERO_LINE_COLOR,
            linestyle=":",
            linewidth=1,
            zorder=0,
        )

        # ------------------------------------------------------------
        # Labels
        # ------------------------------------------------------------

        ax.set_title(f"Signed delay spectrum — SPW {key[0]}")
        if log_scale:
            ax.set_yscale("log")
            ax.set_ylabel(r"$log(|\Delta^2(\tau)|) \propto log(|\tau|^3 P(\tau))$")
        else:
            ax.set_ylabel(r"$|\Delta^2(\tau)| \propto |\tau|^3 P(\tau)$")
        ax.set_xlabel(r"|\tau|$ [ns]")

        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="upper left")

        fig.tight_layout()

        if out_dir:
            output_file = out_dir / f"delta2_tau_signed_multiple_spw_{key[0]}.png"
            fig.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"Saved: {output_file}")
            plt.close()
        else:
            plt.show()

    psc_unfiltered._close()
    if eor_pspec_file:
        psc_eor._close()
    for psc in psc_delays:
        psc._close()


def plot_signal_amplification_vs_bandwidth(
    list_pspec_info,
    output_folder = None,
    weight_mode="count",
):
    """
    Plot leakage score against spectral window bandwidth.

    Parameters
    ----------
    list_pspec_info : 1D array
        Array containing [bl_pair_name, unfiltered_sum_pspec, filtered_sum_pspec, hw_ns]
    
    output_folder : Path, optional
        Path to output place for plots

    weight_mode : String, optional
        "count" => Number of bins filtered > unfiltered\n
        "diff" => Sum(filtered - unfiltered) where filtered > unfiltered\n
        "ratio" => Sum(filtered / unfiltered) where filtered > unfiltered\n
    """

    widths_sorted = []
    scores_sorted_all_bl = []
    spw_sorted = []

    for i in range(len(list_pspec_info)):
        bl_pair = list_pspec_info[i][0]
        unfiltered_sum_pspec = list_pspec_info[i][1]
        filtered_sum_pspec = list_pspec_info[i][2]
        hw_ns = list_pspec_info[i][3]
        spw_indices, scores = helpers.calculate_leakage_scores(
            filtered_sum_pspec,
            unfiltered_sum_pspec,
            hw_ns,
            weight_mode=weight_mode,
        )

        band_widths = helpers.get_band_widths(BAND_STR)
        widths_for_plot = [band_widths[i] for i in spw_indices]

        # Sort by bandwidth.
        order = np.argsort(widths_for_plot)

        if len(widths_sorted) == 0:
            widths_sorted = [widths_for_plot[i] for i in order]
            spw_sorted = [spw_indices[i] for i in order]

        scores_sorted = [scores[i] for i in order]
        scores_sorted_all_bl.append(scores_sorted)

    avg_scores = np.average(scores_sorted_all_bl, axis = 0)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        widths_sorted,
        avg_scores,
        marker="o",
        linestyle="-",
        color="tab:red",
    )

    for w, s, spw in zip(
        widths_sorted,
        avg_scores,
        spw_sorted,
    ):
        ax.annotate(
            str(spw),
            (w, s),
            textcoords="offset points",
            xytext=(5, 5),
        )

    ax.set_xlabel("Spectral window bandwidth (MHz)")

    ylabel_map = {
        "count": "Number of bins where filtered signal > unfiltered signal",
        "ratio": "Amplification-weighted score (Σ filtered/unfiltered, filtered>unfiltered)",
        "diff": "Excess-power-weighted score (Σ filtered-unfiltered, filtered>unfiltered)",
    }

    ax.set_ylabel(ylabel_map[weight_mode])
    ax.set_title(
        f"Avg (over baselines) Signal loss ({weight_mode}-weighted) vs. spectral window bandwidth"
    )

    fig.tight_layout()

    if output_folder:
        out_path = (
            output_folder
            / f"avg_signal_loss_{weight_mode}_vs_bandwidth.png"
        )

        fig.savefig(out_path, dpi=150)

    return out_path


def plot_signal_amplification_vs_spw(
    list_pspec_info,
    output_folder = None,
    weight_mode = "count",
):
    """
    Plot average leakage score over baselines against spectral window index.

    Parameters
    ----------
    list_pspec_info : 1D array
        Array containing [bl_pair_name, unfiltered_sum_pspec, filtered_sum_pspec, hw_ns]
    
    output_folder : Path, optional
        Path to output place for plots

    weight_mode : String, optional
        "count" => Number of bins filtered > unfiltered\n
        "diff" => Sum(filtered - unfiltered) where filtered > unfiltered\n
        "ratio" => Sum(filtered / unfiltered) where filtered > unfiltered\n
    """

    scores_sorted_all_bl = []
    spw_sorted = []

    for i in range(len(list_pspec_info)):
        bl_pair = list_pspec_info[i][0]
        unfiltered_sum_pspec = list_pspec_info[i][1]
        filtered_sum_pspec = list_pspec_info[i][2]
        hw_ns = list_pspec_info[i][3]

        spw_indices, scores = helpers.calculate_leakage_scores(
            filtered_sum_pspec,
            unfiltered_sum_pspec,
            hw_ns,
            weight_mode=weight_mode,
        )

        # Sort by spectral window index.
        order = np.argsort(spw_indices)

        if len(spw_sorted) == 0:
            spw_sorted = [spw_indices[i] for i in order]

        scores_sorted = [scores[i] for i in order]
        scores_sorted_all_bl.append(scores_sorted)

    # Average score for each SPW over all baselines.
    avg_scores = np.average(scores_sorted_all_bl, axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        spw_sorted,
        avg_scores,
        marker="o",
        linestyle="-",
        color="tab:red",
    )

    for spw, s in zip(spw_sorted, avg_scores):
        ax.annotate(
            str(spw),
            (spw, s),
            textcoords="offset points",
            xytext=(5, 5),
        )

    ax.set_xlabel("Spectral window index")

    ylabel_map = {
        "count": "Number of bins where filtered signal > unfiltered signal",
        "ratio": "Amplification-weighted score (Σ filtered/unfiltered, filtered>unfiltered)",
        "diff": "Excess-power-weighted score (Σ filtered-unfiltered, filtered>unfiltered)",
    }

    ax.set_ylabel(ylabel_map[weight_mode])
    ax.set_title(
        f"Avg (over baselines) Signal loss ({weight_mode}-weighted) "
        "vs. spectral window index"
    )

    ax.set_xticks(spw_sorted)

    fig.tight_layout()

    if output_folder:
        out_path = (
            output_folder
            / f"avg_signal_loss_{weight_mode}_vs_spw.png"
        )

        fig.savefig(out_path, dpi=150)

    return out_path