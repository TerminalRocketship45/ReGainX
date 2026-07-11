"""
Quick look: avg muscle fatigue (MF) per STEP across 10 consecutive tasks,
with the exoskeleton assisting vs. no assistance (same high MF=0.70 start,
fatigue persisted across tasks). Shows how fatigue evolves DURING assistance.
Real simulation only; reuses the validated carryover helpers.
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_carryover_experiment import (
    build_envs, start_task, save_state,
    INIT_MF, FORCE_SCALE, ACT_SLOWDOWN, STEPS_PER_TASK)

N_EP = 10
OUT = "carryover_figure"
A_COLOR, B_COLOR = "#D6604D", "#2166AC"


def rollout(assisted, exo_env, exo_policy, seed=0):
    u = exo_env.base_env.unwrapped
    np.random.seed(seed)
    n = exo_env.n_muscles
    MF = np.full(n, INIT_MF)
    split = np.random.uniform(0, 1, n)
    state = {"MF": MF, "MA": (1 - MF) * split, "MR": (1 - MF) * (1 - split)}

    timeline, ep_bounds, ep_stats = [], [], []
    g = 0
    for ep in range(N_EP):
        start_task(exo_env, state)
        ep_bounds.append(g)
        mf0 = float(np.mean(u.muscle_fatigue.MF))
        lstm, es = None, np.ones((1,), bool)
        obs = exo_env._build_obs(exo_env._current_raw_obs())
        for _ in range(STEPS_PER_TASK):
            if assisted:
                a, lstm = exo_policy.predict(obs, state=lstm, episode_start=es,
                                             deterministic=True)
                es = np.zeros((1,), bool)
            else:
                a = np.zeros(exo_env.action_space.shape, np.float32)
            obs, _, d, t, _ = exo_env.step(a)
            timeline.append(float(np.mean(u.muscle_fatigue.MF)))
            g += 1
        mf1 = float(np.mean(u.muscle_fatigue.MF))
        state = save_state(exo_env)
        ep_stats.append((ep + 1, mf0, mf1, mf0 - mf1))
    return np.array(timeline), ep_bounds, ep_stats


def main():
    exo_env, healthy_env, healthy_policy, exo_policy = build_envs()
    mf_B, bounds, statsB = rollout(True, exo_env, exo_policy)    # assisted
    mf_A, _, statsA = rollout(False, exo_env, exo_policy)        # no assist

    # ---- metrics ----
    def summarize(tl, stats, label):
        drops = [s[3] for s in stats]
        print(f"\n[{label}]  start MF={tl[0]:.4f}  end MF={tl[-1]:.4f}  "
              f"net change={tl[-1]-tl[0]:+.4f} ({100*(tl[-1]-tl[0])/tl[0]:+.1f}%)")
        print(f"  mean within-episode change (mf_end-mf_start per ep, "
              f"negative=decrease): {np.mean([s[2]-s[1] for s in stats]):+.4f}")
        print(f"  per-episode start->end MF:")
        for ep, m0, m1, drop in stats:
            print(f"    ep {ep:2d}: {m0:.4f} -> {m1:.4f}  (change {m1-m0:+.4f})")

    summarize(mf_B, statsB, "WITH exoskeleton assistance")
    summarize(mf_A, statsA, "NO assistance")

    # ---- CSV ----
    with open(os.path.join(OUT, "fatigue_during_assist.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["global_step", "avg_MF_assisted", "avg_MF_no_assist"])
        for i in range(len(mf_B)):
            w.writerow([i, mf_B[i], mf_A[i]])

    # ---- figure ----
    plt.rcParams.update({"font.family": "serif", "font.size": 10,
                         "savefig.dpi": 300, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(mf_B))
    ax.plot(x, mf_A, color=A_COLOR, lw=1.8, ls="--",
            label="No assistance (torque = 0)")
    ax.plot(x, mf_B, color=B_COLOR, lw=2.0,
            label="With exoskeleton assistance")
    for b in bounds[1:]:
        ax.axvline(b, color="#cccccc", lw=0.7, zorder=0)
    ax.axhline(INIT_MF, color="#888888", ls=":", lw=1.0)
    ax.text(len(mf_B) * 0.995, INIT_MF + 0.004, f"start MF = {INIT_MF:.2f}",
            ha="right", va="bottom", fontsize=7.5, color="#666666")
    ax.set_xlabel(f"Simulation step ({N_EP} consecutive tasks × {STEPS_PER_TASK} steps)")
    ax.set_ylabel("Average muscle fatigue (MF)")
    ax.set_title("Muscle fatigue during exoskeleton assistance vs. none\n"
                 "(moderate-severe PD, fatigue persisted across tasks)",
                 fontsize=10.5)
    ax.legend(fontsize=8.5, loc="center left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig_fatigue_during_assist.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    exo_env.close(); healthy_env.close()
    print("\nwrote fig_fatigue_during_assist.png/pdf + fatigue_during_assist.csv")


if __name__ == "__main__":
    main()
