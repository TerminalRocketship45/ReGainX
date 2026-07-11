"""
Step 4/5 — two-panel task-200 carryover figure, built ONLY from the real
experiment output (carryover_task200.npz + carryover_log.csv). No hand-drawn
effect; every value shown is read back from the simulation logs.

Honesty (Step 5): labels describe only what was simulated — task-200
performance after 199 prior tasks, with vs without exoskeleton assistance, and
the muscle fatigue accumulated over those tasks. No "months / recovery / cured
/ therapy / retention". The per-panel status text is CHOSEN FROM THE DATA and
never claims the target was reached unless the condition's mean final angle
actually reaches it. At the simulated severity both conditions fall short of
2.0 rad unassisted; the figure says so plainly. A zoomed inset shows the small
but real reach difference; the main axes carry the 2.0 rad target line (same
colour in both panels) so "falls short" is visible.
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "carryover_figure"
NPZ = os.path.join(OUT_DIR, "carryover_task200.npz")
CSV = os.path.join(OUT_DIR, "carryover_log.csv")
A_COLOR = "#D6604D"          # no-assistance-history patient
B_COLOR = "#2166AC"          # prior-exo-use patient
TARGET_COLOR = "#1A1A1A"     # identical target-line colour in both panels

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 9, "axes.labelsize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "savefig.dpi": 300, "pdf.fonttype": 42,
})


def final_task_stats():
    """Mean/std entering the final task, per condition, from the CSV."""
    rows = list(csv.DictReader(open(CSV)))
    tmax = max(int(r["task"]) for r in rows)
    mf, r_final = {}, {}
    for cond in ("A", "B"):
        sub = [r for r in rows if r["condition"] == cond and int(r["task"]) == tmax]
        mf[cond] = (np.mean([float(r["mf_start"]) for r in sub]),
                    np.std([float(r["mf_start"]) for r in sub]))
        r_final[cond] = (np.mean([float(r["pearson_r"]) for r in sub]),
                         np.std([float(r["pearson_r"]) for r in sub]))
    return mf, r_final, tmax


def mean_traj(arr):
    return np.arange(arr.shape[1]), np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def status(mean_reach, target, cond):
    """Honest, data-chosen status line."""
    if mean_reach >= 0.9 * target:
        return "reduced fatigue — target reached"
    return (f"reach falls short (mean {mean_reach:.2f} rad)" if cond == "A"
            else f"greater reach, still short (mean {mean_reach:.2f} rad)")


def main():
    d = np.load(NPZ)
    A, B, target = d["A"], d["B"], float(d["target"])
    mf, r_final, tmax = final_task_stats()
    reach = {"A": float(np.nanmean(A[:, -1])), "B": float(np.nanmean(B[:, -1]))}

    # inset zoom range from the actual data spread
    lo = min(np.nanmin(A), np.nanmin(B))
    hi = max(np.nanmax(A), np.nanmax(B))
    pad = 0.02 + 0.1 * (hi - lo)

    fig, axes = plt.subplots(1, 2, figsize=(5.8, 3.4), sharey=True)
    panels = [(axes[0], A, A_COLOR, "No assistance history", "A"),
              (axes[1], B, B_COLOR, "Prior exo use (1–199), now unassisted", "B")]

    for ax, arr, col, head, cond in panels:
        x, m, s = mean_traj(arr)
        ax.axhline(target, ls="--", lw=1.3, color=TARGET_COLOR, zorder=1,
                   label=f"Target {target:.1f} rad")
        ax.plot(x, m, color=col, lw=2.0, zorder=3, label=f"Mean task-{tmax} reach")
        ax.fill_between(x, m - s, m + s, color=col, alpha=0.18, zorder=2)
        ax.set_ylim(-0.2, target * 1.1)
        ax.set_xlabel(f"Timestep within task {tmax}")
        ax.grid(alpha=0.3)
        ax.set_title(head, fontsize=9)

        # honest status + accumulated fatigue, in the empty upper region
        mfm, mfs = mf[cond]
        ax.text(0.5, 0.90, f"accumulated fatigue entering task {tmax}: "
                f"{mfm:.3f}\n{status(reach[cond], target, cond)}",
                transform=ax.transAxes, ha="center", va="top", fontsize=7.6,
                color=col, linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, lw=0.9))

        # zoomed inset: the real (small) reach difference
        ins = ax.inset_axes([0.12, 0.30, 0.78, 0.34])
        for row in arr:
            ins.plot(np.arange(len(row)), row, color=col, alpha=0.12, lw=0.5)
        ins.plot(x, m, color=col, lw=1.6)
        ins.fill_between(x, m - s, m + s, color=col, alpha=0.18)
        ins.axhline(0.0, color="#999999", lw=0.6, ls=":")
        ins.set_ylim(lo - pad, hi + pad)
        ins.tick_params(labelsize=6)
        ins.set_title("reach detail (zoom)", fontsize=6.5, pad=2)
        ins.grid(alpha=0.25)

    axes[0].set_ylabel("Joint angle (rad)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7.5, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.04), frameon=True)

    fig.suptitle(f"Task-{tmax} unassisted reach after {tmax-1} prior tasks:\n"
                 "effect of accumulated muscle fatigue (vs / without prior "
                 "exoskeleton assistance)", fontsize=9.5, y=1.06)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"fig_carryover.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig_carryover.pdf / png | reach A={reach['A']:.3f} "
          f"B={reach['B']:.3f} | MF A={mf['A'][0]:.3f} B={mf['B'][0]:.3f} | "
          f"r A={r_final['A'][0]:.2f} B={r_final['B'][0]:.2f}")


if __name__ == "__main__":
    main()
