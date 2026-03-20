import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── CVRP Loader (.vrp format) ────────────────────────────────────
def load_cvrp(filepath):
    with open(filepath, "r") as f:
        lines = [l.strip() for l in f.readlines()]

    data = {
        "name"    : "",
        "dimension": 0,
        "capacity": 0,
        "coords"  : [],
        "demands" : [],
        "depot"   : 1
    }

    section = None
    for line in lines:
        if not line or line == "EOF":
            continue
        if line.startswith("NAME"):
            data["name"] = line.split(":")[1].strip()
        elif line.startswith("DIMENSION"):
            data["dimension"] = int(line.split(":")[1].strip())
        elif line.startswith("CAPACITY"):
            data["capacity"] = int(line.split(":")[1].strip())
        elif line == "NODE_COORD_SECTION":
            section = "coords"
        elif line == "DEMAND_SECTION":
            section = "demands"
        elif line == "DEPOT_SECTION":
            section = "depot"
        elif section == "coords":
            parts = line.split()
            if len(parts) == 3:
                data["coords"].append({
                    "id": int(parts[0]),
                    "x" : float(parts[1]),
                    "y" : float(parts[2])
                })
        elif section == "demands":
            parts = line.split()
            if len(parts) == 2:
                data["demands"].append({
                    "id"    : int(parts[0]),
                    "demand": int(parts[1])
                })
        elif section == "depot":
            parts = line.split()
            if parts and parts[0].lstrip("-").isdigit():
                val = int(parts[0])
                if val > 0:
                    data["depot"] = val

    coords_df  = pd.DataFrame(data["coords"])
    demands_df = pd.DataFrame(data["demands"])
    customers  = pd.merge(coords_df, demands_df, on="id")
    dist       = build_distance_matrix(customers[["x","y"]].values)
    depot_idx  = data["depot"] - 1

    return {
        "name"       : data["name"],
        "type"       : "CVRP",
        "capacity"   : data["capacity"],
        "n_nodes"    : len(customers),
        "n_customers": len(customers) - 1,
        "depot_idx"  : depot_idx,
        "customers"  : customers,
        "dist_matrix": dist
    }

# ── VRPTW Loader (Homberger format) ─────────────────────────────
def load_vrptw(filepath):
    with open(filepath, "r") as f:
        lines = [l.strip() for l in f.readlines()]

    name         = lines[0].strip()
    vehicle_line = lines[4].split()
    num_vehicles = int(vehicle_line[0])
    capacity     = int(vehicle_line[1])

    customers = []
    for line in lines[9:]:
        parts = line.split()
        if len(parts) == 7:
            customers.append({
                "id"          : int(parts[0]),
                "x"           : float(parts[1]),
                "y"           : float(parts[2]),
                "demand"      : float(parts[3]),
                "ready_time"  : float(parts[4]),
                "due_date"    : float(parts[5]),
                "service_time": float(parts[6])
            })

    df   = pd.DataFrame(customers)
    dist = build_distance_matrix(df[["x","y"]].values)

    return {
        "name"        : name,
        "type"        : "VRPTW",
        "num_vehicles": num_vehicles,
        "capacity"    : capacity,
        "n_nodes"     : len(df),
        "n_customers" : len(df) - 1,
        "depot_idx"   : 0,
        "customers"   : df,
        "dist_matrix" : dist
    }

# ── Distance matrix ──────────────────────────────────────────────
def build_distance_matrix(coords):
    n    = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i][j] = round(np.sqrt(
                (coords[i][0] - coords[j][0])**2 +
                (coords[i][1] - coords[j][1])**2
            ), 2)
    return dist

# ── Load multiple CVRP instances ─────────────────────────────────
def load_all_cvrp(folder, max_instances=10):
    instances = {}
    files     = sorted([f for f in os.listdir(folder) if f.endswith(".vrp")])
    print(f"Found {len(files)} .vrp files — loading first {max_instances}")
    for fname in files[:max_instances]:
        try:
            inst = load_cvrp(os.path.join(folder, fname))
            instances[fname] = inst
            print(f"  {fname:30s} | Nodes: {inst['n_nodes']} | Cap: {inst['capacity']}")
        except Exception as e:
            print(f"  FAILED: {fname} — {e}")
    return instances

# ── Load multiple VRPTW instances ────────────────────────────────
def load_all_vrptw(folder, max_instances=10):
    instances = {}
    files = sorted([f for f in os.listdir(folder) if f.endswith(".TXT")])
    print(f"Found {len(files)} VRPTW files — loading first {max_instances}")
    for fname in files[:max_instances]:
        try:
            inst = load_vrptw(os.path.join(folder, fname))
            instances[fname] = inst
            print(f"  {fname:20s} | Nodes: {inst['n_nodes']} | "
                  f"Vehicles: {inst['num_vehicles']} | Cap: {inst['capacity']}")
        except Exception as e:
            print(f"  FAILED: {fname} — {e}")
    return instances

# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    CVRP_FOLDER  = r"G:\RESEARCH\supply_chain_project\Transportation\XML"
    VRPTW_FOLDER = r"G:\RESEARCH\supply_chain_project\Transportation\homberger_1000_customer_instances"

    print("=" * 60)
    print("Phase 2 — VRP Data Loader")
    print("=" * 60)

    # Test CVRP
    print("\n[1] CVRP — XML instances")
    print("-" * 60)
    cvrp_file = os.path.join(CVRP_FOLDER, "XML100_1111_01.vrp")
    cvrp      = load_cvrp(cvrp_file)
    print(f"Name        : {cvrp['name']}")
    print(f"Type        : {cvrp['type']}")
    print(f"Nodes       : {cvrp['n_nodes']} (depot + {cvrp['n_customers']} customers)")
    print(f"Capacity    : {cvrp['capacity']}")
    print(f"Depot index : {cvrp['depot_idx']}")
    print(f"Dist matrix : {cvrp['dist_matrix'].shape}")
    print(f"\nFirst 5 nodes:")
    print(cvrp["customers"].head().to_string(index=False))

    # Test VRPTW
    print("\n[2] VRPTW — Homberger instances")
    print("-" * 60)
    vrptw_file = os.path.join(VRPTW_FOLDER, "C1_10_1.TXT")
    vrptw      = load_vrptw(vrptw_file)
    print(f"Name        : {vrptw['name']}")
    print(f"Type        : {vrptw['type']}")
    print(f"Nodes       : {vrptw['n_nodes']} (depot + {vrptw['n_customers']} customers)")
    print(f"Vehicles    : {vrptw['num_vehicles']}")
    print(f"Capacity    : {vrptw['capacity']}")
    print(f"Dist matrix : {vrptw['dist_matrix'].shape}")
    print(f"\nFirst 5 nodes:")
    print(vrptw["customers"].head().to_string(index=False))

    # Save outputs
    os.makedirs("../results/model_scores", exist_ok=True)
    cvrp["customers"].to_csv("../results/model_scores/cvrp_sample.csv",  index=False)
    vrptw["customers"].to_csv("../results/model_scores/vrptw_sample.csv", index=False)

    # Load multiple
    print("\n[3] Loading multiple CVRP instances")
    print("-" * 60)
    all_cvrp = load_all_cvrp(CVRP_FOLDER, max_instances=5)

    print("\n[4] Loading multiple VRPTW instances")
    print("-" * 60)
    all_vrptw = load_all_vrptw(VRPTW_FOLDER, max_instances=5)

    print("\n" + "=" * 60)
    print(f"CVRP  instances loaded : {len(all_cvrp)}")
    print(f"VRPTW instances loaded : {len(all_vrptw)}")
    print("Data loader ready!")
    print("=" * 60)