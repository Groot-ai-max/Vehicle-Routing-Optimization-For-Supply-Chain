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

def routes_to_chromosome(routes, depot):
    chrom = []
    for route in routes:
        for node in route:
            if node != depot:
                chrom.append(node)
    return chrom

# ── Neighborhood moves ───────────────────────────────────────────
def get_neighbors(chromosome, n_neighbors=20):
    neighbors = []
    n         = len(chromosome)
    for _ in range(n_neighbors):
        move_type = random.choice(["swap", "insert", "reverse"])
        chrom     = chromosome[:]

        if move_type == "swap":
            i, j       = random.sample(range(n), 2)
            chrom[i], chrom[j] = chrom[j], chrom[i]
            move       = ("swap", i, j)

        elif move_type == "insert":
            i     = random.randint(0, n-1)
            j     = random.randint(0, n-1)
            node  = chrom.pop(i)
            chrom.insert(j, node)
            move  = ("insert", i, j)

        else:  # reverse
            i, j  = sorted(random.sample(range(n), 2))
            chrom[i:j+1] = chrom[i:j+1][::-1]
            move  = ("reverse", i, j)

        neighbors.append((chrom, move))
    return neighbors

# ── Tabu Search ──────────────────────────────────────────────────
def tabu_search(inst, max_iter=200, tabu_tenure=15, n_neighbors=30):
    dist     = inst["dist_matrix"]
    demands  = inst["customers"]["demand"].values
    capacity = inst["capacity"]
    depot    = inst["depot_idx"]
    nodes    = [i for i in range(inst["n_nodes"]) if i != depot]

    # Initial solution — random
    current       = random.sample(nodes, len(nodes))
    current_score = sum(
        route_distance(r, dist)
        for r in split_into_routes(current, demands, capacity, depot)
    )

    best       = current[:]
    best_score = current_score
    tabu_list  = []
    history    = []

    print(f"  Iter   0 | Best: {best_score:.2f}")

    for iteration in range(1, max_iter + 1):
        neighbors     = get_neighbors(current, n_neighbors)
        best_neighbor = None
        best_nb_score = float("inf")
        best_move     = None

        for chrom, move in neighbors:
            score = sum(
                route_distance(r, dist)
                for r in split_into_routes(chrom, demands, capacity, depot)
            )
            # Accept if not tabu OR aspiration criterion
            if move not in tabu_list or score < best_score:
                if score < best_nb_score:
                    best_nb_score = score
                    best_neighbor = chrom[:]
                    best_move     = move

        if best_neighbor is None:
            break

        current       = best_neighbor
        current_score = best_nb_score

        # Update tabu list
        tabu_list.append(best_move)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

        # Update best
        if current_score < best_score:
            best_score = current_score
            best       = current[:]

        history.append(best_score)
        if iteration % 40 == 0:
            print(f"  Iter {iteration:3d} | Best: {best_score:.2f}")

    best_routes = split_into_routes(best, demands, capacity, depot)
    return best_routes, round(best_score, 2), history

# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    CVRP_FOLDER  = r"G:\RESEARCH\supply_chain_project\Transportation\XML"
    VRPTW_FOLDER = r"G:\RESEARCH\supply_chain_project\Transportation\homberger_1000_customer_instances"
    OUT_SCORES   = r"G:\RESEARCH\supply_chain_project\Transportation\results\model_scores"
    os.makedirs(OUT_SCORES, exist_ok=True)

    results = []

    # ── CVRP ────────────────────────────────────────────────────
    print("=" * 60)
    print("Tabu Search — CVRP")
    print("=" * 60)

    cvrp_files = sorted([f for f in os.listdir(CVRP_FOLDER)
                         if f.endswith(".vrp")])[:5]

    for fname in cvrp_files:
        inst = load_cvrp(os.path.join(CVRP_FOLDER, fname))
        print(f"\n{fname}")
        t0 = time.time()
        routes, dist_ts, history = tabu_search(
            inst, max_iter=200, tabu_tenure=15, n_neighbors=30
        )
        elapsed = round(time.time() - t0, 3)
        print(f"  Final   | dist={dist_ts:.2f} | "
              f"vehicles={len(routes)} | time={elapsed}s")
        results.append({
            "instance"      : fname,
            "type"          : "CVRP",
            "method"        : "Tabu Search",
            "total_distance": dist_ts,
            "num_vehicles"  : len(routes),
            "time_sec"      : elapsed
        })

    # ── VRPTW ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Tabu Search — VRPTW")
    print("=" * 60)

    vrptw_files = sorted([f for f in os.listdir(VRPTW_FOLDER)
                          if f.endswith(".TXT")])[:3]

    for fname in vrptw_files:
        inst = load_vrptw(os.path.join(VRPTW_FOLDER, fname))
        print(f"\n{fname}")
        t0 = time.time()
        routes, dist_ts, history = tabu_search(
            inst, max_iter=100, tabu_tenure=10, n_neighbors=20
        )
        elapsed = round(time.time() - t0, 3)
        print(f"  Final   | dist={dist_ts:.2f} | "
              f"vehicles={len(routes)} | time={elapsed}s")
        results.append({
            "instance"      : fname,
            "type"          : "VRPTW",
            "method"        : "Tabu Search",
            "total_distance": dist_ts,
            "num_vehicles"  : len(routes),
            "time_sec"      : elapsed
        })

    # Save
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_SCORES, "tabu_results.csv"), index=False)
    print("\n" + "=" * 60)
    print(df.to_string(index=False))
    print("\nSaved to results/model_scores/tabu_results.csv")
    print("Day 5 complete!")