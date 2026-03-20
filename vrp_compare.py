import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUT_SCORES = r"G:\RESEARCH\supply_chain_project\Transportation\results\model_scores"
OUT_FIGS   = r"G:\RESEARCH\supply_chain_project\Transportation\results\figs"

# Load all results
baseline  = pd.read_csv(os.path.join(OUT_SCORES, "baseline_results.csv"))
ortools   = pd.read_csv(os.path.join(OUT_SCORES, "ortools_results.csv"))
genetic   = pd.read_csv(os.path.join(OUT_SCORES, "genetic_results.csv"))
tabu      = pd.read_csv(os.path.join(OUT_SCORES, "tabu_results.csv"))
aco       = pd.read_csv(os.path.join(OUT_SCORES, "aco_results.csv"))
hybrid    = pd.read_csv(os.path.join(OUT_SCORES, "hybrid_results.csv"))
pointer   = pd.read_csv(os.path.join(OUT_SCORES, "pointer_net_results.csv"))

# ── Build unified comparison table ──────────────────────────────
rows = []

# Baseline methods
for _, r in baseline.iterrows():
    rows.append({
        "instance": r["instance"],
        "type"    : r["type"],
        "method"  : r["method"],
        "distance": r["total_distance"],
        "vehicles": r["num_vehicles"],
        "time_sec": r["time_sec"]
    })

# OR-Tools
for _, r in ortools.iterrows():
    rows.append({
        "instance": r["instance"],
        "type"    : r["type"],
        "method"  : "OR-Tools",
        "distance": r["total_distance"],
        "vehicles": r["num_vehicles"],
        "time_sec": r["time_sec"]
    })

# GA
for _, r in genetic.iterrows():
    rows.append({
        "instance": r["instance"],
        "type"    : r["type"],
        "method"  : "Genetic Algorithm",
        "distance": r["total_distance"],
        "vehicles": r["num_vehicles"],
        "time_sec": r["time_sec"]
    })

# Tabu
for _, r in tabu.iterrows():
    rows.append({
        "instance": r["instance"],
        "type"    : r["type"],
        "method"  : "Tabu Search",
        "distance": r["total_distance"],
        "vehicles": r["num_vehicles"],
        "time_sec": r["time_sec"]
    })

# ACO
for _, r in aco.iterrows():
    rows.append({
        "instance": r["instance"],
        "type"    : r["type"],
        "method"  : "ACO",
        "distance": r["total_distance"],
        "vehicles": r["num_vehicles"],
        "time_sec": r["time_sec"]
    })

# Pointer Network
for _, r in pointer.iterrows():
    rows.append({
        "instance": r["instance"],
        "type"    : r["type"],
        "method"  : "Pointer Network",
        "distance": r["total_distance"],
        "vehicles": r["num_vehicles"],
        "time_sec": r["time_sec"]
    })

# Hybrid
for _, r in hybrid.iterrows():
    rows.append({
        "instance": r["instance"],
        "type"    : "CVRP",
        "method"  : "Hybrid PPO+2opt",
        "distance": r["dist_hybrid"],
        "vehicles": r["num_vehicles"],
        "time_sec": r["time_sec"]
    })

df_all = pd.DataFrame(rows)

# ── CVRP Summary ─────────────────────────────────────────────────
print("=" * 70)
print("PHASE 2 — COMPLETE BENCHMARK RESULTS")
print("=" * 70)

cvrp_df = df_all[df_all["type"] == "CVRP"].copy()
cvrp_summary = cvrp_df.groupby("method")["distance"].mean().reset_index()
cvrp_summary = cvrp_summary.sort_values("distance").reset_index(drop=True)
cvrp_summary["rank"] = range(1, len(cvrp_summary) + 1)

print("\n[CVRP] Average distance across 5 instances:")
print(cvrp_summary.to_string(index=False))

# ── VRPTW Summary ────────────────────────────────────────────────
vrptw_df = df_all[df_all["type"] == "VRPTW"].copy()
if len(vrptw_df) > 0:
    vrptw_summary = vrptw_df.groupby("method")["distance"].mean().reset_index()
    vrptw_summary = vrptw_summary.sort_values("distance").reset_index(drop=True)
    vrptw_summary["rank"] = range(1, len(vrptw_summary) + 1)
    print("\n[VRPTW] Average distance across 3 instances:")
    print(vrptw_summary.to_string(index=False))

# ── Gap analysis ────────────────────────────────────────────────
print("\n[CVRP] Gap vs Hybrid PPO+2opt (best method):")
best_dist = cvrp_summary[cvrp_summary["method"] == "Hybrid PPO+2opt"]["distance"].values
if len(best_dist) > 0:
    best = best_dist[0]
    for _, row in cvrp_summary.iterrows():
        gap = round(100 * (row["distance"] - best) / best, 2)
        print(f"  {row['method']:25s} | avg_dist={row['distance']:8.2f} | gap={gap:+.2f}%")

# ── Plot 1: Method comparison bar chart ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# CVRP bar chart
methods  = cvrp_summary["method"].tolist()
dists    = cvrp_summary["distance"].tolist()
colors   = ["#1D9E75" if m == "Hybrid PPO+2opt" else
            "#3B8BD4" if m == "OR-Tools" else
            "#888780" for m in methods]

axes[0].barh(methods, dists, color=colors, edgecolor="white")
axes[0].set_xlabel("Avg Total Distance")
axes[0].set_title("CVRP — Method Comparison\n(lower is better)")
axes[0].axvline(x=dists[0], color="#1D9E75", linestyle="--",
                alpha=0.5, label="Best (Hybrid)")
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis="x")

# CVRP instance-level comparison
pivot = cvrp_df[cvrp_df["method"].isin([
    "Hybrid PPO+2opt", "OR-Tools",
    "ACO", "Greedy+2opt"
])].pivot_table(index="instance", columns="method",
                values="distance", aggfunc="mean")

if not pivot.empty:
    pivot.plot(kind="bar", ax=axes[1], width=0.7)
    axes[1].set_title("CVRP — Per Instance Comparison")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Total Distance")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(os.path.join(OUT_FIGS, "phase2_comparison.png"), dpi=150)
plt.close()
print(f"\nSaved: results/figs/phase2_comparison.png")

# ── Plot 2: Improvement over baseline ───────────────────────────
hybrid_data = hybrid.copy()
hybrid_data["improvement"] = hybrid_data["improvement_pct"]

plt.figure(figsize=(8, 5))
bars = plt.bar(range(len(hybrid_data)),
               hybrid_data["improvement"],
               color="#1D9E75", edgecolor="white")
plt.xticks(range(len(hybrid_data)),
           [f.replace("XML100_", "").replace(".vrp", "")
            for f in hybrid_data["instance"]],
           rotation=20)
plt.ylabel("Improvement (%)")
plt.title("Hybrid PPO+2opt — Improvement over PPO alone")
plt.axhline(y=hybrid_data["improvement"].mean(),
            color="red", linestyle="--",
            label=f"Avg: {hybrid_data['improvement'].mean():.1f}%")
plt.legend()
plt.grid(True, alpha=0.3, axis="y")
for bar, val in zip(bars, hybrid_data["improvement"]):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.5,
             f"{val:.1f}%", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIGS, "hybrid_improvement.png"), dpi=150)
plt.close()
print(f"Saved: results/figs/hybrid_improvement.png")

# ── Save final table ─────────────────────────────────────────────
df_all.to_csv(os.path.join(OUT_SCORES, "phase2_final_results.csv"), index=False)
cvrp_summary.to_csv(os.path.join(OUT_SCORES, "phase2_cvrp_summary.csv"), index=False)

print("\n" + "=" * 70)
print("Phase 2 complete!")
print(f"Total methods compared : {df_all['method'].nunique()}")
print(f"Total instances        : {df_all['instance'].nunique()}")
print(f"Best method (CVRP)     : {cvrp_summary.iloc[0]['method']}")
print(f"Best avg distance      : {cvrp_summary.iloc[0]['distance']:.2f}")
print("=" * 70)