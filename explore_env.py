"""Script to explore myosuite environment API."""
import numpy as np
from myosuite.utils import gym

print("=== Healthy env ===")
env = gym.make("myoElbowPose1D6MRandom-v0")
print("Obs space:", env.observation_space)
print("Act space:", env.action_space)
result = env.reset()
print("Reset type:", type(result))
if isinstance(result, tuple):
    print("Tuple len:", len(result))
    obs = result[0]
    info = result[1]
    print("Obs shape:", obs.shape)
    print("Info type:", type(info))
else:
    obs = result
    print("Obs shape:", obs.shape)
env.close()

print("\n=== Fatigue Exo env ===")
env2 = gym.make("myoFatiElbowPose1D6MExoRandom-v0")
print("Obs space:", env2.observation_space)
print("Act space:", env2.action_space)
result2 = env2.reset()
print("Reset type:", type(result2))
if isinstance(result2, tuple):
    print("Tuple len:", len(result2))
    obs2 = result2[0]
    print("Obs shape:", obs2.shape)
else:
    obs2 = result2
    print("Obs shape:", obs2.shape)

unwrapped = env2.unwrapped
print("\n=== Unwrapped attributes ===")
print("Has muscle_fatigue:", hasattr(unwrapped, "muscle_fatigue"))
if hasattr(unwrapped, "muscle_fatigue"):
    mf = unwrapped.muscle_fatigue
    print("MF type:", type(mf))
    print("Has MA:", hasattr(mf, "MA"))
    print("Has MR:", hasattr(mf, "MR"))
    print("Has MF:", hasattr(mf, "MF"))
    if hasattr(mf, "MA"):
        print("MA shape:", mf.MA.shape)
        print("MF shape:", mf.MF.shape)

print("Has target_jnt_range:", hasattr(unwrapped, "target_jnt_range"))
if hasattr(unwrapped, "target_jnt_range"):
    print("target_jnt_range:", unwrapped.target_jnt_range)
    print("target_jnt_range shape:", unwrapped.target_jnt_range.shape)

print("Has target_jnt_value:", hasattr(unwrapped, "target_jnt_value"))
print("Has target_type:", hasattr(unwrapped, "target_type"))
print("Has update_target:", hasattr(unwrapped, "update_target"))
print("Has set_fatigue_reset_random:", hasattr(unwrapped, "set_fatigue_reset_random"))

print("\nHas sim:", hasattr(unwrapped, "sim"))
if hasattr(unwrapped, "sim"):
    sim = unwrapped.sim
    print("Sim type:", type(sim))
    print("Has model:", hasattr(sim, "model"))
    if hasattr(sim, "model"):
        model = sim.model
        print("Has actuator_gear:", hasattr(model, "actuator_gear"))
        if hasattr(model, "actuator_gear"):
            print("actuator_gear shape:", model.actuator_gear.shape)
        print("Has actuator_dynprm:", hasattr(model, "actuator_dynprm"))
        if hasattr(model, "actuator_dynprm"):
            print("actuator_dynprm shape:", model.actuator_dynprm.shape)

print("\nHas get_obs:", hasattr(unwrapped, "get_obs"))
print("Has get_obs_dict:", hasattr(unwrapped, "get_obs_dict"))
if hasattr(unwrapped, "get_obs_dict"):
    obs_dict = unwrapped.get_obs_dict(unwrapped.sim)
    print("obs_dict keys:", list(obs_dict.keys()))
    if "pose_err" in obs_dict:
        pe = obs_dict["pose_err"]
        print("pose_err type:", type(pe))
        print("pose_err shape/value:", pe)

env2.close()
print("\nDone.")
