import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from vrp_data_loader import load_cvrp, load_vrptw
from vrp_baseline import greedy_solution, apply_two_opt, route_distance, total_distance
from vrp_rl_env import VRPEnv
from vrp_rl_agent import train_ppo, evaluate_ppo
from stable_baselines3 import PPO

# ── 2-opt on single route ────────────────────────────────────────
def two_opt_route(route, dist):
    best     = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                if route_distance(new_route, dist) < route_distance(best, dist):
                    best     = new_route
                    improved = True
    return best

# ── Extract routes from PPO policy ──────────────────────────────
def ppo_construct_routes(model, inst, n_trials=5):
    env      = VRPEnv(inst)
    best_dist   = float("inf")
    best_routes = None

    for _ in range(n_trials):
        obs, _  = env.reset()
        done    = False
        steps   = 0
        while not done and steps < env.max_steps:
            action, _ = model.predict(obs, deterministic=False)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done   = terminated or truncated
            steps += 1
        routes, dist = env.get_solution()
        if dist < best_dist and len(routes) > 0:
            best_dist   = dist
            best_routes = routes

    # Fallback to greedy if PPO fails
    if not best_routes:
        demands = inst["customers"]["demand"].values
        best_routes, best_dist = greedy_solution(
            inst["dist_matrix"], demands,
            inst["capacity"], inst["depot_idx"]
        )
    return best_routes, best_dist

# ── Hybrid: PPO construct + 2-opt refine ────────────────────────
def hybrid_ppo_2opt(inst, ppo_model, n_trials=5):
    dist     = inst["dist_matrix"]
    demands  = inst["customers"]["demand"].values
    capacity = inst["capacity"]
    depot    = inst["depot_idx"]

    # Step 1: PPO constructs initial routes
    t0 = time.time()
    routes_ppo, dist_ppo = ppo_construct_routes(ppo_model, inst, n_trials)
    t_ppo = round(time.time() - t0, 3)

    # Step 2: 2-opt refines each route
    t1 = time.time()
    refined_routes = []
    for route in routes_ppo:
        if len(route) > 3:
            refined = two_opt_route(route, dist)
        else:
            refined = route
        refined_routes.append(refined)
    t_refine = round(time.time() - t1, 3)

    dist_hybrid = total_distance(refined_routes, dist)
    improvement = round(100 * (dist_ppo - dist_hybrid) / max(dist_ppo, 1e-8), 2)

    return {
        "routes_ppo"    : routes_ppo,
        "routes_hybrid" : refined_routes,
        "dist_ppo"      : dist_ppo,
        "dist_hybrid"   : dist_hybrid,
        "improvement_pct": improvement,
        "n_vehicles"    : len(refined_routes),
        "t_ppo"         : t_ppo,
        "t_refine"      : t_refine,
        "t_total"       : round(t_ppo + t_refine, 3)
    }

# ── Hybrid: Greedy construct + 2-opt (strong baseline) ──────────
def hybrid_greedy_2opt(inst):
    dist    = inst["dist_matrix"]
    demands = inst["customers"]["demand"].values
    t0      = time.time()
    routes_g, dist_g = greedy_solution(
        dist, demands, inst["capacity"], inst["depot_idx"]
    )
    routes_2, dist_2 = apply_two_opt(routes_g, dist)
    elapsed = round(time.time() - t0, 3)
    return routes_2, dist_2, elapsed

# ── Plot route comparison ────────────────────────────────────────
def plot_comparison(inst, routes_before, routes_after,
                    title_before, title_after, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    df        = inst["customers"]
    colors    = plt.cm.tab20.colors

    for ax, routes, title in zip(axes,
                                  [routes_before, routes_after],
                                  [title_before, title_after]):
        for idx, route in enumerate(routes):
            if len(route) < 2:
                continue
            xs = df.iloc[route]["x"].values
            ys = df.iloc[route]["y"].values
            ax.plot(xs, ys, "-o", color=colors[idx % len(colors)],
                    markersize=3, linewidth=1, alpha=0.7)
        depot = df.iloc[inst["depot_idx"]]
        ax.scatter(depot["x"], depot["y"],
                   c="red", s=200, marker="*", zorder=5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, alpha=0.3)

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

    results    = []
    cvrp_files = sorted([f for f in os.listdir(CVRP_FOLDER)
                         if f.endswith(".vrp")])[:5]

    print("=" * 60)
    print("Hybrid VRP Solver — PPO + 2-opt")
    print("=" * 60)

    for fname in cvrp_files:
        inst = load_cvrp(os.path.join(CVRP_FOLDER, fname))
        print(f"\nInstance: {fname}")

        # Train PPO
        print("  Training PPO...")
        model_path = os.path.join(OUT_MODELS, fname.replace(".vrp","_ppo"))
        try:
            ppo_model = PPO.load(model_path, env=VRPEnv(inst))
            print("  Loaded existing PPO model")
        except:
            ppo_model, _, _ = train_ppo(inst,
                                        total_timesteps=30000,
                                        save_path=model_path)

        # Hybrid PPO + 2-opt
        print("  Running Hybrid PPO + 2-opt...")
        hybrid = hybrid_ppo_2opt(inst, ppo_model, n_trials=10)

        # Greedy + 2-opt for comparison
        _, dist_g2opt, t_g2opt = hybrid_greedy_2opt(inst)

        print(f"  PPO only       : dist={hybrid['dist_ppo']:.2f}")
        print(f"  Hybrid PPO+2opt: dist={hybrid['dist_hybrid']:.2f} | "
              f"improvement={hybrid['improvement_pct']}% | "
              f"time={hybrid['t_total']}s")
        print(f"  Greedy+2opt    : dist={dist_g2opt:.2f}")

        # Plot
        if hybrid["routes_ppo"] and hybrid["routes_hybrid"]:
            plot_comparison(
                inst,
                hybrid["routes_ppo"],
                hybrid["routes_hybrid"],
                f"PPO Routes — {fname}",
                f"Hybrid PPO+2opt — {fname}",
                os.path.join(OUT_FIGS,
                             fname.replace(".vrp", "_hybrid.png"))
            )

        results.append({
            "instance"       : fname,
            "type"           : "CVRP",
            "method"         : "Hybrid PPO+2opt",
            "dist_ppo"       : hybrid["dist_ppo"],
            "dist_hybrid"    : hybrid["dist_hybrid"],
            "dist_greedy_2opt": dist_g2opt,
            "improvement_pct": hybrid["improvement_pct"],
            "num_vehicles"   : hybrid["n_vehicles"],
            "time_sec"       : hybrid["t_total"]
        })

    # Save
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_SCORES, "hybrid_results.csv"), index=False)

    print("\n" + "=" * 60)
    print("HYBRID RESULTS SUMMARY")
    print("=" * 60)
    print(df[["instance", "dist_ppo", "dist_hybrid",
              "dist_greedy_2opt", "improvement_pct",
              "num_vehicles"]].to_string(index=False))
    print("\nSaved to results/model_scores/hybrid_results.csv")
    print("Day 10 complete!")