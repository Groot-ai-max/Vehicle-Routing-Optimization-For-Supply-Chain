import os
import time
import random
import numpy as np
import pandas as pd
from vrp_data_loader import load_cvrp, load_vrptw

# ── Route helpers ────────────────────────────────────────────────
def route_distance(route, dist):
    return sum(dist[route[i]][route[i+1]] for i in range(len(route)-1))

def total_distance(routes, dist):
    return round(sum(route_distance(r, dist) for r in routes), 2)

def split_into_routes(chromosome, demands, capacity, depot):
    routes       = []
    route        = [depot]
    route_demand = 0
    for node in chromosome:
        if route_demand + demands[node] <= capacity:
            route.append(node)
            route_demand += demands[node]
        else:
            route.append(depot)
            routes.append(route)
            route        = [depot, node]
            route_demand = demands[node]
    route.append(depot)
    routes.append(route)
    return routes

# ── Ant Colony Optimization ──────────────────────────────────────
def ant_colony_optimization(inst, n_ants=20, n_iterations=100,
                             alpha=1.0, beta=2.0,
                             evaporation=0.5, Q=100):
    dist     = inst["dist_matrix"]
    demands  = inst["customers"]["demand"].values
    capacity = inst["capacity"]
    depot    = inst["depot_idx"]
    n        = inst["n_nodes"]
    nodes    = [i for i in range(n) if i != depot]

    # Initialize pheromones
    pheromone = np.ones((n, n)) * 0.1
    # Heuristic = 1/distance
    with np.errstate(divide="ignore", invalid="ignore"):
        heuristic = np.where(dist > 0, 1.0 / dist, 0)

    best_routes = None
    best_dist   = float("inf")
    history     = []

    print(f"  Iter   0 | Best: {best_dist:.2f}")

    for iteration in range(1, n_iterations + 1):
        all_routes = []
        all_dists  = []

        for ant in range(n_ants):
            unvisited    = nodes[:]
            chromosome   = []
            current      = depot

            while unvisited:
                # Calculate probabilities
                probs = []
                for j in unvisited:
                    tau  = pheromone[current][j] ** alpha
                    eta  = heuristic[current][j] ** beta
                    probs.append(tau * eta)

                total = sum(probs)
                if total == 0:
                    next_node = random.choice(unvisited)
                else:
                    probs     = [p / total for p in probs]
                    next_node = random.choices(unvisited, weights=probs)[0]

                chromosome.append(next_node)
                unvisited.remove(next_node)
                current = next_node

            routes    = split_into_routes(chromosome, demands, capacity, depot)
            ant_dist  = total_distance(routes, dist)
            all_routes.append(routes)
            all_dists.append(ant_dist)

            if ant_dist < best_dist:
                best_dist   = ant_dist
                best_routes = routes

        # Evaporate pheromones
        pheromone *= (1 - evaporation)

        # Deposit pheromones — best ant only
        best_idx = np.argmin(all_dists)
        for route in all_routes[best_idx]:
            for i in range(len(route) - 1):
                a, b = route[i], route[i+1]
                pheromone[a][b] += Q / all_dists[best_idx]
                pheromone[b][a] += Q / all_dists[best_idx]

        history.append(best_dist)
        if iteration % 20 == 0:
            print(f"  Iter {iteration:3d} | Best: {best_dist:.2f}")

    return best_routes, round(best_dist, 2), history

# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    CVRP_FOLDER  = r"G:\RESEARCH\supply_chain_project\Transportation\XML"
    VRPTW_FOLDER = r"G:\RESEARCH\supply_chain_project\Transportation\homberger_1000_customer_instances"
    OUT_SCORES   = r"G:\RESEARCH\supply_chain_project\Transportation\results\model_scores"
    os.makedirs(OUT_SCORES, exist_ok=True)

    results = []

    # ── CVRP ────────────────────────────────────────────────────
    print("=" * 60)
    print("Ant Colony Optimization — CVRP")
    print("=" * 60)

    cvrp_files = sorted([f for f in os.listdir(CVRP_FOLDER)
                         if f.endswith(".vrp")])[:5]

    for fname in cvrp_files:
        inst = load_cvrp(os.path.join(CVRP_FOLDER, fname))
        print(f"\n{fname}")
        t0 = time.time()
        routes, dist_aco, history = ant_colony_optimization(
            inst, n_ants=20, n_iterations=100,
            alpha=1.0, beta=2.0, evaporation=0.5
        )
        elapsed = round(time.time() - t0, 3)
        print(f"  Final   | dist={dist_aco:.2f} | "
              f"vehicles={len(routes)} | time={elapsed}s")
        results.append({
            "instance"      : fname,
            "type"          : "CVRP",
            "method"        : "ACO",
            "total_distance": dist_aco,
            "num_vehicles"  : len(routes),
            "time_sec"      : elapsed
        })

    # ── VRPTW ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Ant Colony Optimization — VRPTW")
    print("=" * 60)

    vrptw_files = sorted([f for f in os.listdir(VRPTW_FOLDER)
                          if f.endswith(".TXT")])[:3]

    for fname in vrptw_files:
        inst = load_vrptw(os.path.join(VRPTW_FOLDER, fname))
        print(f"\n{fname}")
        t0 = time.time()
        routes, dist_aco, history = ant_colony_optimization(
            inst, n_ants=15, n_iterations=50,
            alpha=1.0, beta=2.0, evaporation=0.5
        )
        elapsed = round(time.time() - t0, 3)
        print(f"  Final   | dist={dist_aco:.2f} | "
              f"vehicles={len(routes)} | time={elapsed}s")
        results.append({
            "instance"      : fname,
            "type"          : "VRPTW",
            "method"        : "ACO",
            "total_distance": dist_aco,
            "num_vehicles"  : len(routes),
            "time_sec"      : elapsed
        })

    # Save
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_SCORES, "aco_results.csv"), index=False)
    print("\n" + "=" * 60)
    print(df.to_string(index=False))
    print("\nSaved to results/model_scores/aco_results.csv")
    print("Day 6 complete!")