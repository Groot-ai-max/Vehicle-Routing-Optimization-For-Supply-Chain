import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from vrp_data_loader import load_cvrp, load_vrptw
from vrp_rl_env import VRPEnv

# ── Training callback ────────────────────────────────────────────
class TrainingCallback(BaseCallback):
    def __init__(self, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.rewards    = []

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            mean_reward = np.mean([ep["r"] for ep in self.model.ep_info_buffer
                                   if "r" in ep] or [0])
            self.rewards.append(mean_reward)
            if self.verbose:
                print(f"  Step {self.n_calls:6d} | "
                      f"Mean reward: {mean_reward:.4f}")
        return True

# ── Train PPO ────────────────────────────────────────────────────
def train_ppo(inst, total_timesteps=50000, save_path=None):
    env      = VRPEnv(inst)
    callback = TrainingCallback(check_freq=5000, verbose=1)

    model = PPO(
        "MlpPolicy", env,
        learning_rate    = 3e-4,
        n_steps          = 512,
        batch_size       = 64,
        n_epochs         = 10,
        gamma            = 0.99,
        gae_lambda       = 0.95,
        clip_range       = 0.2,
        ent_coef         = 0.01,
        verbose          = 0,
        policy_kwargs    = dict(net_arch=[256, 256])
    )

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    elapsed = round(time.time() - t0, 2)

    if save_path:
        model.save(save_path)
        print(f"  Model saved: {save_path}")

    return model, callback.rewards, elapsed

# ── Evaluate PPO ─────────────────────────────────────────────────
def evaluate_ppo(model, inst, n_episodes=5):
    env      = VRPEnv(inst)
    all_dist = []

    for ep in range(n_episodes):
        obs, _   = env.reset()
        done     = False
        steps    = 0

        while not done and steps < env.max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done   = terminated or truncated
            steps += 1

        routes, dist = env.get_solution()
        all_dist.append(dist)

    return {
        "mean_distance": round(np.mean(all_dist), 2),
        "best_distance": round(np.min(all_dist), 2),
        "std_distance" : round(np.std(all_dist), 2),
        "n_routes"     : len(routes)
    }

# ── Plot training curve ──────────────────────────────────────────
def plot_training_curve(rewards, title, save_path):
    plt.figure(figsize=(8, 4))
    plt.plot(rewards, color="steelblue", linewidth=2)
    plt.xlabel("Checkpoint (x5000 steps)")
    plt.ylabel("Mean Reward")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")

# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    CVRP_FOLDER = r"G:\RESEARCH\supply_chain_project\Transportation\XML"
    OUT_SCORES  = r"G:\RESEARCH\supply_chain_project\Transportation\results\model_scores"
    OUT_FIGS    = r"G:\RESEARCH\supply_chain_project\Transportation\results\figs"
    OUT_MODELS  = r"G:\RESEARCH\supply_chain_project\Transportation\results\models"
    os.makedirs(OUT_SCORES, exist_ok=True)
    os.makedirs(OUT_FIGS,   exist_ok=True)
    os.makedirs(OUT_MODELS, exist_ok=True)

    results     = []
    cvrp_files  = sorted([f for f in os.listdir(CVRP_FOLDER)
                          if f.endswith(".vrp")])[:3]

    print("=" * 60)
    print("PPO Agent Training — CVRP")
    print("=" * 60)

    for fname in cvrp_files:
        inst = load_cvrp(os.path.join(CVRP_FOLDER, fname))
        print(f"\nTraining on: {fname}")

        model_path = os.path.join(OUT_MODELS,
                                  fname.replace(".vrp", "_ppo"))
        model, rewards, elapsed = train_ppo(
            inst,
            total_timesteps = 50000,
            save_path       = model_path
        )

        # Plot training curve
        plot_training_curve(
            rewards,
            title     = f"PPO Training — {fname}",
            save_path = os.path.join(OUT_FIGS,
                        fname.replace(".vrp", "_ppo_curve.png"))
        )

        # Evaluate
        eval_result = evaluate_ppo(model, inst, n_episodes=5)
        print(f"  Eval  | mean_dist={eval_result['mean_distance']:.2f} | "
              f"best_dist={eval_result['best_distance']:.2f} | "
              f"routes={eval_result['n_routes']} | "
              f"train_time={elapsed}s")

        results.append({
            "instance"      : fname,
            "type"          : "CVRP",
            "method"        : "PPO",
            "total_distance": eval_result["best_distance"],
            "num_vehicles"  : eval_result["n_routes"],
            "time_sec"      : elapsed
        })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_SCORES, "ppo_results.csv"), index=False)

    print("\n" + "=" * 60)
    print(df.to_string(index=False))
    print("\nSaved to results/model_scores/ppo_results.csv")
    print("Day 8 complete!")