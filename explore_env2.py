"""Further exploration of myosuite API."""
import numpy as np
from myosuite.utils import gym
import inspect

env = gym.make("myoFatiElbowPose1D6MExoRandom-v0")
unwrapped = env.unwrapped

# Check reset signature
print("=== reset signature ===")
sig = inspect.signature(unwrapped.reset)
print("Params:", list(sig.parameters.keys()))

# Try reset with fatigue_reset
print("\n=== Reset without keyword ===")
try:
    result = env.reset()
    print("OK, type:", type(result))
except Exception as e:
    print("Error:", e)

print("\n=== Reset with fatigue_reset=True ===")
try:
    result = env.reset(fatigue_reset=True)
    print("OK, type:", type(result))
except Exception as e:
    print("Error:", e)

# Check set_fatigue_reset_random
print("\n=== set_fatigue_reset_random ===")
sig2 = inspect.signature(unwrapped.set_fatigue_reset_random)
print("Params:", list(sig2.parameters.keys()))

# Try step
env.reset()
action = env.action_space.sample()
print("\n=== Step ===")
step_result = env.step(action)
print("Step result len:", len(step_result))
print("Types:", [type(x).__name__ for x in step_result])

# Check muscle fatigue attrs
mf = unwrapped.muscle_fatigue
print("\n=== Muscle fatigue attrs ===")
print(dir(mf))

# Check observation construction
print("\n=== get_obs ===")
obs = unwrapped.get_obs()
print("get_obs shape:", obs.shape)
obs_dict = unwrapped.get_obs_dict(unwrapped.sim)
print("obs_dict:", {k: getattr(v, 'shape', v) for k, v in obs_dict.items()})

env.close()

# Now check what the frozen policy (healthy) sees
print("\n=== Healthy env deeper ===")
henv = gym.make("myoElbowPose1D6MRandom-v0")
hobs, _ = henv.reset()
print("Healthy obs shape:", hobs.shape)
hobs_dict = henv.unwrapped.get_obs_dict(henv.unwrapped.sim)
print("Healthy obs_dict:", {k: getattr(v, 'shape', v) for k, v in hobs_dict.items()})
henv.close()
