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

def fitness(chromosome, demands, capacity, depot, dist):
    routes = split_into_routes(chromosome, demands, capacity, depot)
    return total_distance(routes, dist)

# ── GA Operators ─────────────────────────────────────────────────
def order_crossover(p1, p2):
    n    = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = [None] * n
    child[a:b+1] = p1[a:b+1]
    fill  = [x for x in p2 if x not in child]
    idx   = 0
    for i in range(n):
        if child[i] is None:
            child[i] = fill[idx]
            idx += 1
    return child

def mutate(chromosome, rate=0.02):
    chrom = chromosome[:]
    for i in range(len(chrom)):
        if random.random() < rate:
            j         = random.randint(0, len(chrom)-1)
            chrom[i], chrom[j] = chrom[j], chrom[i]
    return chrom

def tournament_select(pop, scores, k=3):
    idxs = random.sample(range(len(pop)), k)
    best = min(idxs, key=lambda i: scores[i])
    return pop[best][:]

# ── Genetic Algorithm ────────────────────────────────────────────
def genetic_algorithm(inst, pop_size=50, generations=100,
                      mutation_rate=0.02, elite_size=5):
    dist     = inst["dist_matrix"]
    demands  = inst["customers"]["demand"].values
    capacity = inst["capacity"]
    depot    = inst["depot_idx"]
    nodes    = [i for i in range(inst["n_nodes"]) if i != depot]

    # Initial population
    population = [random.sample(nodes, len(nodes)) for _ in range(pop_size)]
    scores     = [fitness(c, demands, capacity, depot, dist) for c in population]

    best_score  = min(scores)
    best_chrom  = population[scores.index(best_score)][:]
    history     = []

    print(f"  Gen 0   | Best: {best_score:.2f}")

    for gen in range(1, generations + 1):
        # Elitism
        elite_idx  = sorted(range(len(scores)), key=lambda i: scores[i])[:elite_size]
        new_pop    = [population[i][:] for i in elite_idx]

        # Crossover + mutation
        while len(new_pop) < pop_size:
            p1    = tournament_select(population, scores)
            p2    = tournament_select(population, scores)
            child = order_crossover(p1, p2)
            child = mutate(child, mutation_rate)
            new_pop.append(child)

        population = new_pop
        scores     = [fitness(c, demands, capacity, depot, dist)
                      for c in population]
        gen_best   = min(scores)

        if gen_best < best_score:
            best_score = gen_best
            best_chrom = population[scores.index(gen_best)][:]

        history.append(best_score)
        if gen % 20 == 0:
            print(f"  Gen {gen:3d} | Best: {best_score:.2f}")

    best_routes = split_into_routes(best_chrom, demands, capacity, depot)
    return best_routes, best_score, history

# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    CVRP_FOLDER  = r"G:\RESEARCH\supply_chain_project\Transportation\XML"
    VRPTW_FOLDER = r"G:\RESEARCH\supply_chain_project\Transportation\homberger_1000_customer_instances"
    OUT_SCORES   = r"G:\RESEARCH\supply_chain_project\Transportation\results\model_scores"
    os.makedirs(OUT_SCORES, exist_ok=True)

    results = []

    # ── CVRP ────────────────────────────────────────────────────
    print("=" * 60)
    print("Genetic Algorithm — CVRP")
    print("=" * 60)

    cvrp_files = sorted([f for f in os.listdir(CVRP_FOLDER)
                         if f.endswith(".vrp")])[:5]

    for fname in cvrp_files:
        inst = load_cvrp(os.path.join(CVRP_FOLDER, fname))
        print(f"\n{fname}")
        t0 = time.time()
        routes, dist_ga, history = genetic_algorithm(
            inst, pop_size=50, generations=100, mutation_rate=0.02
        )
        elapsed = round(time.time() - t0, 3)
        print(f"  Final   | dist={dist_ga:.2f} | "
              f"vehicles={len(routes)} | time={elapsed}s")
        results.append({
            "instance"      : fname,
            "type"          : "CVRP",
            "method"        : "Genetic Algorithm",
            "total_distance": dist_ga,
            "num_vehicles"  : len(routes),
            "time_sec"      : elapsed
        })

    # ── VRPTW ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Genetic Algorithm — VRPTW")
    print("=" * 60)

    vrptw_files = sorted([f for f in os.listdir(VRPTW_FOLDER)
                          if f.endswith(".TXT")])[:3]

    for fname in vrptw_files:
        inst = load_vrptw(os.path.join(VRPTW_FOLDER, fname))
        print(f"\n{fname}")
        t0 = time.time()
        routes, dist_ga, history = genetic_algorithm(
            inst, pop_size=30, generations=50, mutation_rate=0.02
        )
        elapsed = round(time.time() - t0, 3)
        print(f"  Final   | dist={dist_ga:.2f} | "
              f"vehicles={len(routes)} | time={elapsed}s")
        results.append({
            "instance"      : fname,
            "type"          : "VRPTW",
            "method"        : "Genetic Algorithm",
            "total_distance": dist_ga,
            "num_vehicles"  : len(routes),
            "time_sec"      : elapsed
        })

    # Save
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_SCORES, "genetic_results.csv"), index=False)
    print("\n" + "=" * 60)
    print(df.to_string(index=False))
    print("\nSaved to results/model_scores/genetic_results.csv")
    print("Day 4 complete!")