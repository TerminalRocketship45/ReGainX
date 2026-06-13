ReGainX -- Research Export Package
====================================
Generated: 2026-06-13
Evaluation: 4,000 episodes per policy
Grid: 4 angle bins x 4 severity quartiles = 16 cells, 250 episodes per cell
Environment: brady+deg (bradykinesia=True, smart_reset=True) for ALL exo policies
Metric definitions:
  Pearson r     = Pearson correlation between impaired+exo joint angle trajectory
                  and a fresh healthy reference trajectory run in parallel
  Goal rate     = fraction of episodes where rwd_dict["solved"] fired at least once
  Reward        = cumulative undiscounted episode reward from MyoSuite pose-error signal
  Severity      = composite score in [0,1]: mean of normalised force_scale drop,
                  activation_slowdown, and muscle fatigue (MF)

Files in this package
---------------------
00_README.txt                         this file
01_all_policies_master_summary.csv    one row per policy, all metrics
02_per_quartile_all_policies.csv      per-severity-quartile breakdown, all policies combined
03_per_angle_all_policies.csv         per-angle-bin breakdown, all policies combined
04_heatmap_pearsonr.csv               4x4 Pearson r heatmap for each policy
04_heatmap_goalrate.csv               4x4 goal rate heatmap for each policy
05_noise_ablation.csv                 RecPPO clean vs noise-trained at sigma 0/0.01/0.05/0.10
06_ablation_gaps.csv                  computed gap tables for all 4 ablations
07_training_curve_stats.csv           final reward, max, convergence TS, plateau flag
08_key_numbers.txt                    human-readable results for paper writing

Policies evaluated
------------------
policy_brady_deg_recurrent     RecurrentPPO  trained on brady+deg  (2M steps intended, 1M logged)
policy_deg_recurrent           RecurrentPPO  trained on deg-only   (1M steps)
policy_brady_deg               MLP PPO       trained on brady+deg  (1M steps)
policy_deg                     MLP PPO       trained on deg-only   (1M steps)
policy_brady_deg_recurrent_noisy  RecurrentPPO  brady+deg + EMG noise domain randomisation
                                              (1.5M initial + 1.5M finetune = 3M total)
no_exo                         zero-torque baseline (impaired patient, no assistance)
healthy_policy                 healthy PPO on healthy env (reference ceiling)
