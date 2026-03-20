import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from vrp_data_loader import load_cvrp, load_vrptw

def plot_instance(inst, title=None, save_path=None):
    df      = inst["customers"]
    depot   = inst["depot_idx"]
    depot_row = df[df["id"] == df.iloc[depot]["id"]]
    customers = df.drop(index=depot)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot customers
    ax.scatter(customers["x"], customers["y"],
               c="steelblue", s=30, zorder=3, label="Customers")

    # Plot depot
    ax.scatter(depot_row["x"], depot_row["y"],
               c="red", s=200, marker="*", zorder=5, label="Depot")

    # Annotate depot
    ax.annotate("Depot",
                xy=(depot_row["x"].values[0], depot_row["y"].values[0]),
                xytext=(10, 10), textcoords="offset points", fontsize=9)

    ax.set_title(title or inst["name"], fontsize=13)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close()

def plot_demand_distribution(inst, save_path=None):
    df = inst["customers"]
    demands = df[df["demand"] > 0]["demand"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Demand histogram
    axes[0].hist(demands, bins=20, color="steelblue", edgecolor="white")
    axes[0].set_title("Demand Distribution")
    axes[0].set_xlabel("Demand")
    axes[0].set_ylabel("Count")

    # Demand scatter on map
    scatter = axes[1].scatter(
        df[df["demand"] > 0]["x"],
        df[df["demand"] > 0]["y"],
        c=demands, cmap="YlOrRd", s=40, zorder=3
    )
    depot_row = df.iloc[inst["depot_idx"]]
    axes[1].scatter(depot_row["x"], depot_row["y"],
                    c="blue", s=200, marker="*", zorder=5, label="Depot")
    plt.colorbar(scatter, ax=axes[1], label="Demand")
    axes[1].set_title("Demand Heatmap")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"{inst['name']} — Demand Analysis", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close()

def plot_time_windows(inst, save_path=None):
    """Only for VRPTW instances"""
    if inst["type"] != "VRPTW":
        print("Time windows only available for VRPTW instances")
        return

    df = inst["customers"].iloc[1:]  # exclude depot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Time window width distribution
    df["tw_width"] = df["due_date"] - df["ready_time"]
    axes[0].hist(df["tw_width"], bins=20, color="teal", edgecolor="white")
    axes[0].set_title("Time Window Width Distribution")
    axes[0].set_xlabel("Window Width")
    axes[0].set_ylabel("Count")

    # Ready time vs due date scatter
    axes[1].scatter(df["ready_time"], df["due_date"],
                    alpha=0.5, c="purple", s=20)
    axes[1].plot([0, df["due_date"].max()],
                 [0, df["due_date"].max()], "r--", alpha=0.3)
    axes[1].set_title("Ready Time vs Due Date")
    axes[1].set_xlabel("Ready Time")
    axes[1].set_ylabel("Due Date")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"{inst['name']} — Time Window Analysis", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close()

if __name__ == "__main__":
    CVRP_FOLDER  = r"G:\RESEARCH\supply_chain_project\Transportation\XML"
    VRPTW_FOLDER = r"G:\RESEARCH\supply_chain_project\Transportation\homberger_1000_customer_instances"
    OUT          = r"G:\RESEARCH\supply_chain_project\Transportation\results\figs"
    os.makedirs(OUT, exist_ok=True)

    print("=" * 55)
    print("Phase 2 — Instance Visualizer")
    print("=" * 55)

    # CVRP plots
    cvrp = load_cvrp(os.path.join(CVRP_FOLDER, "XML100_1111_01.vrp"))
    plot_instance(cvrp,
                  title="CVRP Instance — XML100_1111_01",
                  save_path=os.path.join(OUT, "cvrp_instance.png"))
    plot_demand_distribution(cvrp,
                  save_path=os.path.join(OUT, "cvrp_demand.png"))

    # VRPTW plots
    vrptw = load_vrptw(os.path.join(VRPTW_FOLDER, "C1_10_1.TXT"))
    plot_instance(vrptw,
                  title="VRPTW Instance — C1_10_1",
                  save_path=os.path.join(OUT, "vrptw_instance.png"))
    plot_demand_distribution(vrptw,
                  save_path=os.path.join(OUT, "vrptw_demand.png"))
    plot_time_windows(vrptw,
                  save_path=os.path.join(OUT, "vrptw_time_windows.png"))

    print("\n5 plots saved to results/figs/")
    print("Day 1 complete!")
    print("=" * 55)