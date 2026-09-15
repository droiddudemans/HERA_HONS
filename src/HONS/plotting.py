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


# Fixed, reserved colors used by every plot in this module so the
# "reference" curves are always visually consistent and never get
# reassigned to a filtered curve.
UNFILTERED_COLOR = "black"
EOR_COLOR = "darkorange"
BOUNDARY_COLOR = "dimgray"
ZERO_LINE_COLOR = "lightgray"

_RESERVED_COLORS = [UNFILTERED_COLOR, EOR_COLOR, BOUNDARY_COLOR, ZERO_LINE_COLOR]


def plot_delta2_tau_multiple(
    unfiltered_pspec_file,
    eor_pspec_file,
    delay_pspec_files,
    out_dir,
    delay_hw,
    labels=None,
    unfiltered_label="Unfiltered EOR + FG",
    eor_label="Unfiltered EOR only",
):
    """
    Plot folded/unsigned Delta^2(tau) for multiple delay-filtered
    PSpec files together with the unfiltered signal.

    Negative and positive delays are folded onto |tau| and averaged.

    Parameters
    ----------
    unfiltered_pspec_file : Path
        PSpec file containing the unfiltered EOR + foreground signal.

    eor_pspec_file : Path
        PSpec file containing the unfiltered EOR signal.

    delay_pspec_files : list[Path]
        List of delay-filtered PSpec files to compare.

    out_dir : Path
        Directory where plots are saved.

    delay_hw : float
        Nominal delay-filter half-width in ns.

    labels : list[str], optional
        Legend labels for the delay-filtered curves, one per entry in
        delay_pspec_files. If None, the filenames are used.

    unfiltered_label : str, optional
        Legend label for the unfiltered curve.

    eor_label : str, optional
        Legend label for the unfiltered-EOR-only curve.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    psc_unfiltered = hp.PSpecContainer(unfiltered_pspec_file, mode="r")
    psc_eor = hp.PSpecContainer(eor_pspec_file, mode="r")
    psc_delays = [
        hp.PSpecContainer(delay_pspec_file, mode="r")
        for delay_pspec_file in delay_pspec_files
    ]

    uvp_unfiltered = psc_unfiltered.get_pspec("stokespol", "time_and_interleave_averaged")
    uvp_eor = psc_eor.get_pspec("stokespol", "time_and_interleave_averaged")
    uvp_delays = [
        psc.get_pspec("stokespol", "time_and_interleave_averaged")
        for psc in psc_delays
    ]

    if labels is None:
        labels = [Path(f).stem for f in delay_pspec_files]

    if len(labels) != len(delay_pspec_files):
        raise ValueError("Number of labels must match number of delay_pspec_files.")

    # Guaranteed-distinct colors for the filtered curves only. The
    # unfiltered/eor/boundary colors are fixed constants above and are
    # excluded from this pool, so nothing can ever collide.
    filtered_colors = _get_distinct_colors(
        len(delay_pspec_files), reserved_colors=_RESERVED_COLORS
    )

    print(f"Nominal delay half-width: {delay_hw} ns")

    for key in uvp_delays[0].get_all_keys():

        # ------------------------------------------------------------
        # Delay axis
        # ------------------------------------------------------------

        delays = np.squeeze(uvp_delays[0].get_dlys(key[0]))
        delays_ns = delays * 1e9

        neg = delays_ns < 0
        pos = delays_ns > 0

        tau_neg = np.abs(delays_ns[neg])
        tau_pos = delays_ns[pos]

        # ------------------------------------------------------------
        # Unfiltered
        # ------------------------------------------------------------

        P_unfiltered = np.squeeze(uvp_unfiltered.get_data(key).real)
        delta2_unfiltered_neg = tau_neg**3 * P_unfiltered[neg]
        delta2_unfiltered_pos = tau_pos**3 * P_unfiltered[pos]

        # ------------------------------------------------------------
        # Unfiltered EOR only
        # ------------------------------------------------------------

        P_eor = np.squeeze(uvp_eor.get_data(key).real)
        delta2_eor_neg = tau_neg**3 * P_eor[neg]
        delta2_eor_pos = tau_pos**3 * P_eor[pos]

        # A single min_len based on the delay axis itself, used
        # consistently for every curve on this plot so nothing gets
        # silently truncated to a different length than its neighbors.
        min_len = min(len(tau_neg), len(tau_pos))

        tau = tau_pos[:min_len]

        delta2_unfiltered_avg = (
            delta2_unfiltered_pos[:min_len] + delta2_unfiltered_neg[:min_len]
        ) / 2.0

        delta2_eor_avg = (
            delta2_eor_pos[:min_len] + delta2_eor_neg[:min_len]
        ) / 2.0

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

            delta2_filtered_neg = (tau_neg**3 * P_filtered[neg])[:min_len]
            delta2_filtered_pos = (tau_pos**3 * P_filtered[pos])[:min_len]

            delta2_filtered_avg = (delta2_filtered_pos + delta2_filtered_neg) / 2.0

            ax.plot(
                tau,
                np.abs(delta2_filtered_avg),
                lw=2,
                color=color,
                label=label,
                zorder=2,
            )

        # ------------------------------------------------------------
        # Delay-filter boundary
        # ------------------------------------------------------------

        ax.axvline(
            delay_hw,
            color=BOUNDARY_COLOR,
            linestyle="--",
            linewidth=1.5,
            label=rf"$\tau_\mathrm{{hw}}={delay_hw:.1f}$ ns",
            zorder=1,
        )

        # ------------------------------------------------------------
        # Axes
        # ------------------------------------------------------------

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$|\tau|$ [ns]")
        ax.set_ylabel(r"$|\Delta^2(\tau)| \propto |\tau|^3 P(\tau)$")
        ax.set_title(rf"Delay spectrum $\Delta^2(\tau)$ — SPW {key[0]}")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()

        fig.tight_layout()

        output_file = out_dir / f"delta2_tau_multiple_spw_{key[0]}.png"
        fig.savefig(output_file, dpi=300, bbox_inches="tight")

        plt.show()
        plt.close(fig)

        print(f"Saved: {output_file}")

    psc_unfiltered._close()
    psc_eor._close()
    for psc in psc_delays:
        psc._close()


def plot_delta2_tau_signed_multiple(
    unfiltered_pspec_file,
    eor_pspec_file,
    delay_pspec_files,
    out_dir,
    delay_hw,
    labels=None,
    unfiltered_label="Unfiltered EOR + FG",
    eor_label="Unfiltered EOR only",
):
    """
    Plot Delta^2(tau) against the signed delay axis for multiple
    delay-filtered PSpec files, together with the unfiltered signal.

    Parameters
    ----------
    unfiltered_pspec_file : Path
        PSpec file containing the unfiltered EOR + foreground signal.

    eor_pspec_file : Path
        PSpec file containing the unfiltered EOR signal.

    delay_pspec_files : list[Path]
        List of delay-filtered PSpec files to compare.

    out_dir : Path
        Directory where plots are saved.

    delay_hw : float
        Nominal delay-filter half-width in ns.

    labels : list[str], optional
        Legend labels for the delay-filtered curves, one per entry in
        delay_pspec_files. If None, the filenames are used.

    unfiltered_label : str, optional
        Legend label for the unfiltered curve.

    eor_label : str, optional
        Legend label for the unfiltered-EOR-only curve.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    psc_unfiltered = hp.PSpecContainer(unfiltered_pspec_file, mode="r")
    psc_eor = hp.PSpecContainer(eor_pspec_file, mode="r")
    psc_delays = [
        hp.PSpecContainer(delay_pspec_file, mode="r")
        for delay_pspec_file in delay_pspec_files
    ]

    uvp_unfiltered = psc_unfiltered.get_pspec("stokespol", "time_and_interleave_averaged")
    uvp_eor = psc_eor.get_pspec("stokespol", "time_and_interleave_averaged")
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

    print(f"Nominal delay half-width: {delay_hw} ns")

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

        negative_unfiltered = np.isfinite(delta2_unfiltered) & (delays_ns < 0)
        positive_unfiltered = np.isfinite(delta2_unfiltered) & (delays_ns > 0)

        # ------------------------------------------------------------
        # Unfiltered EOR only signal
        # ------------------------------------------------------------

        P_eor = np.squeeze(uvp_eor.get_data(key).real)
        delta2_eor = np.abs(delays_ns) ** 3 * P_eor

        # BUG FIX: `positive_eor` previously reused `delays_ns < 0`,
        # so it was identical to `negative_eor` and the entire
        # positive-delay half of the EOR-only curve was silently
        # dropped from the plot.
        negative_eor = np.isfinite(delta2_eor) & (delays_ns < 0)
        positive_eor = np.isfinite(delta2_eor) & (delays_ns > 0)

        # ------------------------------------------------------------
        # Plot
        # ------------------------------------------------------------

        fig, ax = plt.subplots(figsize=(10, 6))

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
        # Filtered curves — each gets its own guaranteed-unique color,
        # shared between its negative- and positive-delay halves
        # ------------------------------------------------------------

        for uvp_delay, label, color in zip(uvp_delays, labels, filtered_colors):

            P_filtered = np.squeeze(uvp_delay.get_data(key).real)
            delta2_filtered = np.abs(delays_ns) ** 3 * P_filtered

            negative_filtered = np.isfinite(delta2_filtered) & (delays_ns < 0)
            positive_filtered = np.isfinite(delta2_filtered) & (delays_ns > 0)

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

        # ------------------------------------------------------------
        # Delay-filter boundary
        # ------------------------------------------------------------

        ax.axvline(
            -delay_hw,
            color=BOUNDARY_COLOR,
            linestyle="--",
            linewidth=1.5,
            label=rf"$\pm\tau_\mathrm{{hw}}={delay_hw:.1f}$ ns",
            zorder=1,
        )
        ax.axvline(
            delay_hw,
            color=BOUNDARY_COLOR,
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

        ax.set_xlabel(r"$\tau$ [ns]")
        ax.set_ylabel(r"$|\Delta^2(\tau)| \propto |\tau|^3 P(\tau)$")
        ax.set_title(f"Signed delay spectrum — SPW {key[0]}")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc='upper left')

        fig.tight_layout()

        output_file = out_dir / f"delta2_tau_signed_multiple_spw_{key[0]}.png"
        fig.savefig(output_file, dpi=300, bbox_inches="tight")

        plt.show()
        plt.close(fig)

        print(f"Saved: {output_file}")

    psc_unfiltered._close()
    psc_eor._close()
    for psc in psc_delays:
        psc._close()


#-------------------PLOTTING FUNCTIONS-------------------#

def plot_delta2_tau_signed(
    unfiltered_pspec_file,
    delay_pspec_file,
    out_dir,
    delay_hw,
    eor_pspec_file=None,
):
    """
    Plot Delta^2(tau) against the signed delay axis.

    Negative and positive delays remain on their respective sides
    of zero, while both are shown on the same axes.

    eor_pspec_file is optional. If not provided, only the
    filtered vs. unfiltered comparison is plotted (no EOR-only trace).
    """

    has_eor = eor_pspec_file is not None

    psc_unfiltered = hp.PSpecContainer(unfiltered_pspec_file, mode="r")
    psc_delay = hp.PSpecContainer(delay_pspec_file, mode="r")
    psc_eor = hp.PSpecContainer(eor_pspec_file, mode="r") if has_eor else None

    out_dir.mkdir(parents=True, exist_ok=True)

    uvp_unfiltered = psc_unfiltered.get_pspec(
        "stokespol",
        "time_and_interleave_averaged"
    )

    uvp_delay = psc_delay.get_pspec(
        "stokespol",
        "time_and_interleave_averaged"
    )

    uvp_eor = (
        psc_eor.get_pspec("stokespol", "time_and_interleave_averaged")
        if has_eor else None
    )

    print(f"Nominal delay half-width: {delay_hw} ns")

    for key in uvp_delay.get_all_keys():

        true_hw = helpers.find_true_hw(uvp_delay, key, delay_hw)
        P_unfiltered = np.squeeze(
            uvp_unfiltered.get_data(key).real
        )

        P_fg = np.squeeze(
            uvp_delay.get_data(key).real
        )

        P_eor = np.squeeze(uvp_eor.get_data(key).real) if has_eor else None

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
        delta2_PN = np.abs(delays_ns)**3 * PN
        delta2_eor = np.abs(delays_ns)**3 * P_eor if has_eor else None

        # ------------------------------------------------------------
        # Finite positive values only
        # ------------------------------------------------------------

        valid_fg = (
            np.isfinite(delta2_fg)
        )

        if has_eor:
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

        negative_fg = np.isfinite(delta2_fg) & (delays_ns < 0)
        positive_fg = np.isfinite(delta2_fg) & (delays_ns > 0)

        if has_eor:
            negative_eor = np.isfinite(delta2_eor) & (delays_ns < 0)
            positive_eor = np.isfinite(delta2_eor) & (delays_ns > 0)

        negative_unfiltered = np.isfinite(delta2_unfiltered) & (delays_ns < 0)
        positive_unfiltered = np.isfinite(delta2_unfiltered) & (delays_ns > 0)

        negative_PN = np.isfinite(delta2_PN) & (delays_ns < 0)
        positive_PN = np.isfinite(delta2_PN) & (delays_ns > 0)

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

        # EOR only (optional)
        if has_eor:
            ax.plot(
                delays_ns[negative_eor],
                delta2_eor[negative_eor],
                lw = 2,
                linestyle = "--",
                color = "blue",
                label = rf"$\Delta^2(\tau)$ for EOR only"
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
        # "True" half-width boundaries
        # ------------------------------------------------------------

        ax.axvline(
            true_hw,
            color = 'm',
            linestyle = '--',
            linewidth = 1.5,
            label=fr"$\pm\tau_\mathrm{{'true' hw}}={true_hw:.1f}$ ns"
        )

        ax.axvline(
            -true_hw,
            color = 'm',
            linestyle = '--',
            linewidth = 1.5
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
    if has_eor:
        psc_eor._close()

# Plots delta^2(tau)

def plot_delta2_tau(
    unfiltered_pspec_file,
    delay_pspec_file,
    out_dir,
    delay_hw,
    eor_pspec_file=None,
):
    """
    Plot qualitative
        Delta^2(tau) ~ tau^3 P(tau)

    comparing delay-filtered EOR + foregrounds against
    unfiltered EOR + foregrounds. If eor_pspec_file is provided,
    also overlays the EOR-only trace.

    This is unsigned in tau.

    delay_hw is in nanoseconds.
    """

    has_eor = eor_pspec_file is not None

    psc_unfiltered_sum = hp.PSpecContainer(unfiltered_pspec_file, mode='r')
    psc_delay = hp.PSpecContainer(delay_pspec_file, mode="r")
    psc_eor = hp.PSpecContainer(eor_pspec_file, mode="r") if has_eor else None

    out_dir.mkdir(parents=True, exist_ok=True)

    uvp_unfiltered_sum = psc_unfiltered_sum.get_pspec(
        "stokespol",
        "time_and_interleave_averaged"
    )

    uvp_delay = psc_delay.get_pspec(
        "stokespol",
        "time_and_interleave_averaged"
    )

    uvp_eor = (
        psc_eor.get_pspec("stokespol", "time_and_interleave_averaged")
        if has_eor else None
    )

    print(f"Nominal delay half-width: {delay_hw} ns")

    for key in uvp_delay.get_all_keys():

        P_unfiltered = np.squeeze(
            uvp_unfiltered_sum.get_data(key).real
        )

        P_fg = np.squeeze(
            uvp_delay.get_data(key).real
        )

        P_eor = np.squeeze(uvp_eor.get_data(key).real) if has_eor else None

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

        delta2_unfiltered_neg = tau_neg**3 * P_unfiltered[neg]
        delta2_unfiltered_pos = tau_pos**3 * P_unfiltered[pos]

        if has_eor:
            delta2_eor_neg = tau_neg**3 * P_eor[neg]
            delta2_eor_pos = tau_pos**3 * P_eor[pos]

        len_list = [
            len(tau_neg),
            len(tau_pos),
            len(delta2_fg_neg),
            len(delta2_fg_pos),
            len(PN_neg),
            len(PN_pos),
            len(delta2_unfiltered_neg),
            len(delta2_unfiltered_pos),
        ]
        if has_eor:
            len_list += [len(delta2_eor_neg), len(delta2_eor_pos)]

        min_len = min(len_list)

        tau_neg = tau_neg[:min_len]
        tau_pos = tau_pos[:min_len]

        delta2_fg_neg = delta2_fg_neg[:min_len]
        delta2_fg_pos = delta2_fg_pos[:min_len]

        PN_neg = PN_neg[:min_len]
        PN_pos = PN_pos[:min_len]

        delta2_unfiltered_neg = delta2_unfiltered_neg[:min_len]
        delta2_unfiltered_pos = delta2_unfiltered_pos[:min_len]

        if has_eor:
            delta2_eor_neg = delta2_eor_neg[:min_len]
            delta2_eor_pos = delta2_eor_pos[:min_len]

        delta2_fg_avg = (delta2_fg_pos + delta2_fg_neg) / 2.0
        PN_avg = (PN_neg + PN_pos) / 2.0
        delta2_unfiltered_avg = (delta2_unfiltered_pos + delta2_unfiltered_neg) / 2.0

        if has_eor:
            delta2_eor_avg = (delta2_eor_pos + delta2_eor_neg) / 2.0

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

        if has_eor:
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
    if has_eor:
        psc_eor._close()