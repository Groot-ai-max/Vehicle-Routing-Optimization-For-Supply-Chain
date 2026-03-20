import os
import time
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from vrp_data_loader import load_cvrp, load_vrptw

# ── OR-Tools CVRP Solver ─────────────────────────────────────────
def solve_cvrp(inst, time_limit_sec=30):
    dist     = inst["dist_matrix"]
    demands  = inst["customers"]["demand"].values.tolist()
    capacity = inst["capacity"]
    depot    = inst["depot_idx"]
    n        = inst["n_nodes"]

    # Scale distances to integers
    scale    = 100
    dist_int = [[int(dist[i][j] * scale) for j in range(n)] for i in range(n)]

    manager = pywrapcp.RoutingIndexManager(n, inst.get("num_vehicles", 25), depot)
    routing = pywrapcp.RoutingModel(manager)

    # Distance callback
    def dist_callback(from_idx, to_idx):
        return dist_int[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)]
    transit_cb = routing.RegisterTransitCallback(dist_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

    # Capacity constraint
    def demand_callback(from_idx):
        return int(demands[manager.IndexToNode(from_idx)])
    demand_cb = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_cb, 0, [int(capacity)] * inst.get("num_vehicles", 25),
        True, "Capacity"
    )

    # Search parameters
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.seconds = time_limit_sec

    t0       = time.time()
    solution = routing.SolveWithParameters(params)
    elapsed  = round(time.time() - t0, 3)

    if not solution:
        return None

    routes     = []
    total_dist = 0
    for v in range(inst.get("num_vehicles", 25)):
        idx   = routing.Start(v)
        route = []
        while not routing.IsEnd(idx):
            route.append(manager.IndexToNode(idx))
            idx = solution.Value(routing.NextVar(idx))
        route.append(manager.IndexToNode(idx))
        if len(route) > 2:
            routes.append(route)
            total_dist += sum(
                dist[route[i]][route[i+1]] for i in range(len(route)-1)
            )

    return {
        "routes"         : routes,
        "total_distance" : round(total_dist, 2),
        "num_vehicles"   : len(routes),
        "time_sec"       : elapsed,
        "status"         : "Optimal" if solution else "Failed"
    }

# ── OR-Tools VRPTW Solver ────────────────────────────────────────
def solve_vrptw(inst, time_limit_sec=60):
    dist         = inst["dist_matrix"]
    df           = inst["customers"]
    demands      = df["demand"].values.tolist()
    capacity     = inst["capacity"]
    depot        = inst["depot_idx"]
    n            = inst["n_nodes"]
    num_vehicles = inst["num_vehicles"]

    scale    = 10
    dist_int = [[int(dist[i][j] * scale) for j in range(n)] for i in range(n)]
    tw_starts = [int(df.iloc[i]["ready_time"]   * scale) for i in range(n)]
    tw_ends   = [int(df.iloc[i]["due_date"]      * scale) for i in range(n)]
    svc_times = [int(df.iloc[i]["service_time"]  * scale) for i in range(n)]

    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    # Distance + service time callback
    def time_callback(from_idx, to_idx):
        f = manager.IndexToNode(from_idx)
        t = manager.IndexToNode(to_idx)
        return dist_int[f][t] + svc_times[f]
    time_cb = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(time_cb)

    # Time window dimension
    routing.AddDimension(
        time_cb, 3000 * scale, 3000 * scale, False, "Time"
    )
    time_dim = routing.GetDimensionOrDie("Time")
    for i in range(n):
        idx = manager.NodeToIndex(i)
        time_dim.CumulVar(idx).SetRange(tw_starts[i], tw_ends[i])

    # Capacity constraint
    def demand_cb(from_idx):
        return int(demands[manager.IndexToNode(from_idx)])
    d_cb = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        d_cb, 0, [int(capacity)] * num_vehicles, True, "Capacity"
    )

    # Search params
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.seconds = time_limit_sec

    t0       = time.time()
    solution = routing.SolveWithParameters(params)
    elapsed  = round(time.time() - t0, 3)

    if not solution:
        return None

    routes     = []
    total_dist = 0
    for v in range(num_vehicles):
        idx   = routing.Start(v)
        route = []
        while not routing.IsEnd(idx):
            route.append(manager.IndexToNode(idx))
            idx = solution.Value(routing.NextVar(idx))
        route.append(manager.IndexToNode(idx))
        if len(route) > 2:
            routes.append(route)
            total_dist += sum(
                dist[route[i]][route[i+1]] for i in range(len(route)-1)
            )

    return {
        "routes"        : routes,
        "total_distance": round(total_dist, 2),
        "num_vehicles"  : len(routes),
        "time_sec"      : elapsed,
        "status"        : "Optimal"
    }

# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    CVRP_FOLDER  = r"G:\RESEARCH\supply_chain_project\Transportation\XML"
    VRPTW_FOLDER = r"G:\RESEARCH\supply_chain_project\Transportation\homberger_1000_customer_instances"
    OUT_SCORES   = r"G:\RESEARCH\supply_chain_project\Transportation\results\model_scores"
    os.makedirs(OUT_SCORES, exist_ok=True)

    results = []

    # ── CVRP ────────────────────────────────────────────────────
    print("=" * 60)
    print("OR-Tools CVRP Solver")
    print("=" * 60)

    cvrp_files = sorted([f for f in os.listdir(CVRP_FOLDER)
                         if f.endswith(".vrp")])[:5]

    for fname in cvrp_files:
        inst = load_cvrp(os.path.join(CVRP_FOLDER, fname))
        sol  = solve_cvrp(inst, time_limit_sec=30)

        if sol:
            print(f"{fname:30s} | dist={sol['total_distance']:8.2f} | "
                  f"vehicles={sol['num_vehicles']:3d} | "
                  f"time={sol['time_sec']}s")
            results.append({
                "instance"      : fname,
                "type"          : "CVRP",
                "method"        : "OR-Tools",
                "total_distance": sol["total_distance"],
                "num_vehicles"  : sol["num_vehicles"],
                "time_sec"      : sol["time_sec"]
            })
        else:
            print(f"{fname:30s} | No solution found")

    # ── VRPTW ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("OR-Tools VRPTW Solver")
    print("=" * 60)

    vrptw_files = sorted([f for f in os.listdir(VRPTW_FOLDER)
                          if f.endswith(".TXT")])[:3]

    for fname in vrptw_files:
        inst = load_vrptw(os.path.join(VRPTW_FOLDER, fname))
        sol  = solve_vrptw(inst, time_limit_sec=60)

        if sol:
            print(f"{fname:20s} | dist={sol['total_distance']:8.2f} | "
                  f"vehicles={sol['num_vehicles']:3d} | "
                  f"time={sol['time_sec']}s")
            results.append({
                "instance"      : fname,
                "type"          : "VRPTW",
                "method"        : "OR-Tools",
                "total_distance": sol["total_distance"],
                "num_vehicles"  : sol["num_vehicles"],
                "time_sec"      : sol["time_sec"]
            })
        else:
            print(f"{fname:20s} | No solution found")

    # Save
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_SCORES, "ortools_results.csv"), index=False)
    print("\n" + "=" * 60)
    print(df.to_string(index=False))
    print("\nSaved to results/model_scores/ortools_results.csv")
    print("Day 3 complete!")