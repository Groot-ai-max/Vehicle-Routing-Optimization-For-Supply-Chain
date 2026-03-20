import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vrp_data_loader import load_cvrp, load_vrptw

# ── Greedy Nearest Neighbor ──────────────────────────────────────
def greedy_solution(dist_matrix, demands, capacity, depot_idx=0):
    n         = len(demands)
    visited   = [False] * n
    visited[depot_idx] = True
    routes    = []
    total_dist = 0

    while not all(visited):
        route        = [depot_idx]
        route_demand = 0
        current      = depot_idx

        while True:
            best_dist = float("inf")
            best_node = None

            for j in range(n):
                if not visited[j] and j != depot_idx:
                    if route_demand + demands[j] <= capacity:
                        if dist_matrix[current][j] < best_dist:
                            best_dist = dist_matrix[current][j]
                            best_node = j

            if best_node is None:
                break

            visited[best_node] = True
            route.append(best_node)
            route_demand += demands[best_node]
            total_dist   += dist_matrix[current][best_node]
            current       = best_node

        # Return to depot
        total_dist += dist_matrix[current][depot_idx]
        route.append(depot_idx)
        routes.append(route)

    return routes, round(total_dist, 2)

# ── 2-opt Improvement ────────────────────────────────────────────
def two_opt(route, dist_matrix):
    best     = route[:]
    improved = True

    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                if route_distance(new_route, dist_matrix) < \
                   route_distance(best, dist_matrix):
                    best     = new_route
                    improved = True
    return best

def route_distance(route, dist_matrix):
    return sum(dist_matrix[route[i]][route[i+1]]
               for i in range(len(route) - 1))

def total_distance(routes, dist_matrix):
    return round(sum(route_distance(r, dist_matrix) for r in routes), 2)

def apply_two_opt(routes, dist_matrix):
    improved_routes = []
    total_dist      = 0
    for route in routes:
        better = two_opt(route, dist_matrix)
        improved_routes.append(better)
        total_dist += route_distance(better, dist_matrix)
    return improved_routes, round(total_dist, 2)

# ── Evaluate solution ────────────────────────────────────────────
def evaluate(routes, dist_matrix, demands, capacity):
    total_dist     = 0
    total_vehicles = len(routes)
    feasible       = True

    for route in routes:
        dist         = route_distance(route, dist_matrix)
        route_demand = sum(demands[n] for n in route)
        total_dist  += dist
        if route_demand > capacity:
            feasible = False

    return {
        "total_distance": round(total_dist, 2),
        "num_vehicles"  : total_vehicles,
        "feasible"      : feasible
    }

# ── Plot routes ──────────────────────────────────────────────────
def plot_routes(inst, routes, title, save_path):
    df     = inst["customers"]
    colors = plt.cm.tab20.colors

    fig, ax = plt.subplots(figsize=(10, 8))

    for idx, route in enumerate(routes):
        color = colors[idx % len(colors)]
        xs    = df.iloc[route]["x"].values
        ys    = df.iloc[route]["y"].values
        ax.plot(xs, ys, "-o", color=color,
                markersize=4, linewidth=1.2, alpha=0.8)

    depot = df.iloc[inst["depot_idx"]]
    ax.scatter(depot["x"], depot["y"],
               c="red", s=250, marker="*", zorder=5, label="Depot")

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")

# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    CVRP_FOLDER  = r"G:\RESEARCH\supply_chain_project\Transportation\XML"
    VRPTW_FOLDER = r"G:\RESEARCH\supply_chain_project\Transportation\homberger_1000_customer_instances"
    OUT_FIGS     = r"G:\RESEARCH\supply_chain_project\Transportation\results\figs"
    OUT_SCORES   = r"G:\RESEARCH\supply_chain_project\Transportation\results\model_scores"
    os.makedirs(OUT_FIGS,   exist_ok=True)
    os.makedirs(OUT_SCORES, exist_ok=True)

    results = []

    # ── CVRP Experiment ──────────────────────────────────────────
    print("=" * 60)
    print("CVRP Baseline — XML instances")
    print("=" * 60)

    cvrp_files = sorted([f for f in os.listdir(CVRP_FOLDER)
                         if f.endswith(".vrp")])[:5]

    for fname in cvrp_files:
        inst     = load_cvrp(os.path.join(CVRP_FOLDER, fname))
        demands  = inst["customers"]["demand"].values
        dist     = inst["dist_matrix"]
        cap      = inst["capacity"]
        depot    = inst["depot_idx"]

        t0 = time.time()
        routes_g, dist_g = greedy_solution(dist, demands, cap, depot)
        t_greedy = round(time.time() - t0, 3)
        eval_g   = evaluate(routes_g, dist, demands, cap)

        t0 = time.time()
        routes_2, dist_2 = apply_two_opt(routes_g, dist)
        t_2opt = round(time.time() - t0, 3)
        eval_2 = evaluate(routes_2, dist, demands, cap)

        improvement = round(100*(dist_g - dist_2)/dist_g, 2)

        print(f"\n{fname}")
        print(f"  Greedy     : dist={dist_g:8.2f} | "
              f"vehicles={eval_g['num_vehicles']:3d} | "
              f"time={t_greedy}s")
        print(f"  Greedy+2opt: dist={dist_2:8.2f} | "
              f"vehicles={eval_2['num_vehicles']:3d} | "
              f"time={t_2opt}s | "
              f"improvement={improvement}%")

        results.append({
            "instance"      : fname,
            "type"          : "CVRP",
            "method"        : "Greedy",
            "total_distance": dist_g,
            "num_vehicles"  : eval_g["num_vehicles"],
            "time_sec"      : t_greedy
        })
        results.append({
            "instance"      : fname,
            "type"          : "CVRP",
            "method"        : "Greedy+2opt",
            "total_distance": dist_2,
            "num_vehicles"  : eval_2["num_vehicles"],
            "time_sec"      : t_2opt
        })

    # Plot best CVRP routes
    inst    = load_cvrp(os.path.join(CVRP_FOLDER, cvrp_files[0]))
    demands = inst["customers"]["demand"].values
    dist    = inst["dist_matrix"]
    routes_g, _ = greedy_solution(dist, demands, inst["capacity"], inst["depot_idx"])
    routes_2, _ = apply_two_opt(routes_g, dist)

    plot_routes(inst, routes_g, "CVRP — Greedy Routes",
                os.path.join(OUT_FIGS, "cvrp_greedy_routes.png"))
    plot_routes(inst, routes_2, "CVRP — Greedy + 2-opt Routes",
                os.path.join(OUT_FIGS, "cvrp_2opt_routes.png"))

    # ── VRPTW Experiment ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VRPTW Baseline — Homberger instances")
    print("=" * 60)

    vrptw_files = sorted([f for f in os.listdir(VRPTW_FOLDER)
                          if f.endswith(".TXT")])[:3]

    for fname in vrptw_files:
        inst    = load_vrptw(os.path.join(VRPTW_FOLDER, fname))
        demands = inst["customers"]["demand"].values
        dist    = inst["dist_matrix"]
        cap     = inst["capacity"]
        depot   = inst["depot_idx"]

        t0 = time.time()
        routes_g, dist_g = greedy_solution(dist, demands, cap, depot)
        t_greedy = round(time.time() - t0, 3)
        eval_g   = evaluate(routes_g, dist, demands, cap)

        t0 = time.time()
        routes_2, dist_2 = apply_two_opt(routes_g, dist)
        t_2opt = round(time.time() - t0, 3)
        eval_2 = evaluate(routes_2, dist, demands, cap)

        improvement = round(100*(dist_g - dist_2)/dist_g, 2)

        print(f"\n{fname}")
        print(f"  Greedy     : dist={dist_g:8.2f} | "
              f"vehicles={eval_g['num_vehicles']:3d} | "
              f"time={t_greedy}s")
        print(f"  Greedy+2opt: dist={dist_2:8.2f} | "
              f"vehicles={eval_2['num_vehicles']:3d} | "
              f"time={t_2opt}s | "
              f"improvement={improvement}%")

        results.append({
            "instance"      : fname,
            "type"          : "VRPTW",
            "method"        : "Greedy",
            "total_distance": dist_g,
            "num_vehicles"  : eval_g["num_vehicles"],
            "time_sec"      : t_greedy
        })
        results.append({
            "instance"      : fname,
            "type"          : "VRPTW",
            "method"        : "Greedy+2opt",
            "total_distance": dist_2,
            "num_vehicles"  : eval_2["num_vehicles"],
            "time_sec"      : t_2opt
        })

    # Plot VRPTW routes
    inst    = load_vrptw(os.path.join(VRPTW_FOLDER, vrptw_files[0]))
    demands = inst["customers"]["demand"].values
    dist    = inst["dist_matrix"]
    routes_g, _ = greedy_solution(dist, demands, inst["capacity"], inst["depot_idx"])
    routes_2, _ = apply_two_opt(routes_g, dist)

    plot_routes(inst, routes_g, "VRPTW — Greedy Routes",
                os.path.join(OUT_FIGS, "vrptw_greedy_routes.png"))
    plot_routes(inst, routes_2, "VRPTW — Greedy + 2-opt Routes",
                os.path.join(OUT_FIGS, "vrptw_2opt_routes.png"))

    # ── Save results ─────────────────────────────────────────────
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(OUT_SCORES,
                      "baseline_results.csv"), index=False)

    print("\n" + "=" * 60)
    print("BASELINE SUMMARY")
    print("=" * 60)
    print(df_results.to_string(index=False))
    print("\nSaved to results/model_scores/baseline_results.csv")
    print("Day 2 complete!")