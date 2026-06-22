ReGainX Paper Figures
============================================================

All figures: serif font, seaborn-v0_8-whitegrid style, 300 DPI, 6.5in wide, saved as .png and .pdf.
Consistent color palette across every figure:
  RecurrentPPO brady+deg : #2166AC (deep blue)
  RecurrentPPO deg-only  : #D6604D (red-orange)
  MLP brady+deg          : #4DAC26 (green)
  MLP deg-only           : #B2ABD2 (light purple)
  Noisy RecurrentPPO     : #74ADD1 (light blue)
  No-exoskeleton baseline: #808080 (grey)
  Healthy baseline       : #1A1A1A (near-black)

arxiv.sty: included

Figures (root + figures/ subfolder both contain copies):
------------------------------------------------------------

Figure 1: fig1_pipeline.png / fig1_pipeline.pdf
  Location : Section 4 (Methods) — Figure 1
  Caption  : Three-phase ReGainX pipeline shown with real MuJoCo/MyoSuite renders at a 2.0 rad target: (1) a trained healthy controller reaches the target; (2) the same controller, frozen under bradykinesia + muscular degeneration, stalls short; (3) a trained RecurrentPPO exoskeleton agent assists the frozen patient back to the target.
  Data     : MuJoCo/MyoSuite render of healthy, frozen-patient and exo-assisted rollouts at target 2.0 rad (render_pipeline_frames.py, seed 1)

Figure 2: fig2_pd_modeling.png / fig2_pd_modeling.pdf
  Location : Section 3 (PD Modeling) — Figure 2
  Caption  : Parkinsonian motor-deficit model. Left: muscle activation is reduced by degeneration and further delayed under combined PD. Right: joint-angle trajectories show reduced reach and delayed onset.
  Data     : synthetic illustrative curves (numpy)

Figure 3: fig3_training_curves.png / fig3_training_curves.pdf
  Location : Section 4 (Training) — Figure 3
  Caption  : Training reward curves (rolling mean, window=20, with std band) for the recurrent and MLP policies across training timesteps.
  Data     : loaded logs/*_rewards.csv (rolling mean/std, window=20); policies: RecurrentPPO brady+deg, RecurrentPPO deg-only, Noisy RecurrentPPO, MLP brady+deg, MLP deg-only

Figure 4: fig9_ablation1.png / fig9_ablation1.pdf
  Location : Section 5 (Ablations) — Ablation 1
  Caption  : Ablation 1: per-severity Pearson r (bars) and mean episodic reward (lines) for RecurrentPPO brady+deg vs. the MLP brady+deg baseline across severity quartiles.
  Data     : loaded per_quartile CSVs (RecurrentPPO vs MLP)

Figure 5: fig4_ablation_severity.png / fig4_ablation_severity.pdf
  Location : Section 5 (Ablations) — Figure 4
  Caption  : Ablation 2: per-severity Pearson r (bars) and mean episodic reward (lines) for RecurrentPPO brady+deg vs. deg-only across severity quartiles; at Q4 severe the deg-only reward collapses to 9.26 vs. 382.30 for the combined policy.
  Data     : loaded per_quartile CSVs (r, std, reward)

Figure 6: fig5_heatmap.png / fig5_heatmap.pdf
  Location : Section 5 (Results) — Figure 5
  Caption  : Mean Pearson r across target-angle bins (rows) and severity quartiles (columns) for our combined-trained RecurrentPPO (left) vs. the degeneration-only state-of-the-art baseline (right). Blue scale.
  Data     : loaded per_cell CSVs: brady+deg (ours) vs. deg-only (baseline)

Figure 7: fig6_radian_performance.png / fig6_radian_performance.pdf
  Location : Section 5 (Results) — Figure 6
  Caption  : Trajectory-tracking Pearson r by target-angle bin for four policies with std bands; the LSTM accumulates context and improves at larger angles. Dashed line marks no-exo overall r=0.223.
  Data     : loaded per_radian CSVs (std bands)

Figure 8: fig7_noise_robustness.png / fig7_noise_robustness.pdf
  Location : Section 5 (Robustness) — Figure 7
  Caption  : Noise robustness: clean vs. noisy-trained RecurrentPPO across observation noise sigma; performance changes by only dr=0.011 over the full tested range.
  Data     : hardcoded logged values

Figure 9: fig8_latency.png / fig8_latency.pdf
  Location : Section 6 (Deployment) — Figure 8
  Caption  : Per-step inference latency for each policy (mean +/- std) against the 10-20 ms exoskeleton real-time control budget; all policies run well within budget.
  Data     : hardcoded logged values (latency_bench)

------------------------------------------------------------
Data provenance summary (loaded vs hardcoded):
  fig1_pipeline: MuJoCo/MyoSuite render of healthy, frozen-patient and exo-assisted rollouts at target 2.0 rad (render_pipeline_frames.py, seed 1)
  fig2_pd_modeling: synthetic illustrative curves (numpy)
  fig3_training_curves: loaded logs/*_rewards.csv (rolling mean/std, window=20); policies: RecurrentPPO brady+deg, RecurrentPPO deg-only, Noisy RecurrentPPO, MLP brady+deg, MLP deg-only
  fig9_ablation1: loaded per_quartile CSVs (RecurrentPPO vs MLP)
  fig4_ablation_severity: loaded per_quartile CSVs (r, std, reward)
  fig5_heatmap: loaded per_cell CSVs: brady+deg (ours) vs. deg-only (baseline)
  fig6_radian_performance: loaded per_radian CSVs (std bands)
  fig7_noise_robustness: hardcoded logged values
  fig8_latency: hardcoded logged values (latency_bench)
