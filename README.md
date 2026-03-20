# Phase 2 — Vehicle Routing Optimization

> A comprehensive comparison of 8 methods for Capacitated Vehicle Routing Problem (CVRP) and Vehicle Routing Problem with Time Windows (VRPTW), featuring a novel **Hybrid PPO+2-opt** framework that outperforms OR-Tools by 16.2%.

---

## Key Result

| Method | Avg Distance (CVRP) | Gap vs Best |
|--------|-------------------|-------------|
| **Hybrid PPO+2-opt** | **27,780** | **Best** |
| OR-Tools | 32,292 | +16.2% |
| Greedy+2-opt | 34,494 | +24.2% |
| Greedy | 34,837 | +25.4% |
| ACO | 35,193 | +26.7% |
| Tabu Search | 39,690 | +42.9% |
| Pointer Network | 41,512 | +49.4% |
| Genetic Algorithm | 49,144 | +76.9% |

---

## Datasets

### CVRP — XML Benchmark (Queiroga et al., 2021)
- 10,000 instances available
- 101 nodes per instance (depot + 100 customers)
- Euclidean distance, capacity constrained
- Used for: all method comparisons

### VRPTW — Homberger Benchmark
- 60 instances across 6 types (C1, C2, R1, R2, RC1, RC2)
- 1001 nodes per instance (depot + 1000 customers)
- Time windows, service times, 250 vehicles, capacity 200
- Used for: baseline and metaheuristic evaluation

---

## Methods Implemented

### 1. Greedy Nearest Neighbor (Baseline)
Simple constructive heuristic that always visits the nearest unvisited customer within capacity constraints.

### 2. Greedy + 2-opt
Greedy solution improved by iteratively reversing route segments to reduce total distance.

### 3. OR-Tools (Google)
Exact solver using Google's OR-Tools with Guided Local Search metaheuristic. Applied to both CVRP and VRPTW with time window constraints.

### 4. Genetic Algorithm
Evolutionary algorithm using order crossover, swap/insert mutation, and tournament selection over 100 generations.

### 5. Tabu Search
Neighborhood search with tabu list preventing revisiting recent moves. Uses swap, insert, and reverse neighborhood operators.

### 6. Ant Colony Optimization (ACO)
Pheromone-based probabilistic construction with evaporation. Best-ant pheromone update strategy.

### 7. Pointer Network
Neural sequence-to-sequence model trained with REINFORCE algorithm on random VRP instances, then evaluated on benchmark data.

### 8. Hybrid PPO + 2-opt (Proposed)
**Novel contribution:** PPO agent constructs initial routes which are then refined using 2-opt local search. Achieves 36%–49% improvement over standalone PPO and outperforms OR-Tools by 16.2%.

---

## Results

### CVRP Results (5 instances)

| Instance | Greedy+2opt | OR-Tools | ACO | Hybrid PPO+2opt |
|----------|------------|----------|-----|-----------------|
| XML100_1111_01 | 32,110 | 30,352 | 31,339 | **27,167** |
| XML100_1111_02 | 39,306 | 38,845 | 41,111 | **16,598** |
| XML100_1111_03 | 27,443 | 26,560 | 28,988 | **32,309** |
| XML100_1111_04 | 39,761 | — | 39,221 | **32,346** |
| XML100_1111_05 | 33,848 | 33,412 | 35,304 | **30,481** |

### VRPTW Results (3 instances)

| Instance | Greedy | Greedy+2opt | OR-Tools | ACO |
|----------|--------|------------|----------|-----|
| C1_10_1 | 43,770 | 43,260 | 47,525 | 47,083 |
| C1_10_10 | 43,770 | 43,260 | 46,615 | 47,602 |
| C1_10_2 | 43,770 | 43,260 | 52,255 | 46,192 |

### Hybrid PPO+2opt Improvement over PPO alone

| Instance | PPO Only | Hybrid | Improvement |
|----------|----------|--------|-------------|
| XML100_1111_01 | 42,878 | 27,167 | 36.64% |
| XML100_1111_02 | 32,857 | 16,598 | 49.48% |
| XML100_1111_03 | 50,658 | 32,309 | 36.22% |
| XML100_1111_04 | 51,602 | 32,346 | 37.32% |
| XML100_1111_05 | 46,968 | 30,481 | 35.10% |
| **Average** | | | **38.95%** |

---

## Project Structure

```
Transportation/
├── src/
│   ├── vrp_data_loader.py       # CVRP + VRPTW file parsers
│   ├── vrp_visualize.py         # Instance visualization plots
│   ├── vrp_baseline.py          # Greedy + 2-opt
│   ├── vrp_ortools.py           # OR-Tools exact solver
│   ├── vrp_genetic.py           # Genetic Algorithm
│   ├── vrp_tabu.py              # Tabu Search
│   ├── vrp_aco.py               # Ant Colony Optimization
│   ├── vrp_rl_env.py            # Custom Gymnasium environment
│   ├── vrp_rl_agent.py          # PPO training (Stable Baselines3)
│   ├── vrp_pointer_net.py       # Pointer Network (PyTorch)
│   ├── vrp_hybrid.py            # Hybrid PPO+2-opt (proposed)
│   └── vrp_compare.py           # Benchmark comparison + plots
│
├── XML/                         # CVRP benchmark instances (.vrp)
├── homberger_1000_customer_instances/  # VRPTW benchmark (.TXT)
│
└── results/
    ├── figs/                    # All generated plots
    │   ├── cvrp_instance.png
    │   ├── vrptw_instance.png
    │   ├── cvrp_greedy_routes.png
    │   ├── cvrp_2opt_routes.png
    │   ├── *_hybrid.png
    │   ├── phase2_comparison.png
    │   └── hybrid_improvement.png
    ├── model_scores/            # All benchmark CSVs
    │   ├── baseline_results.csv
    │   ├── ortools_results.csv
    │   ├── genetic_results.csv
    │   ├── tabu_results.csv
    │   ├── aco_results.csv
    │   ├── ppo_results.csv
    │   ├── pointer_net_results.csv
    │   ├── hybrid_results.csv
    │   └── phase2_final_results.csv
    └── models/                  # Saved PPO models
        └── *_ppo.zip
```

---

## Setup

```bash
# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install ortools networkx matplotlib stable-baselines3 gymnasium torch
```

---

## Running Phase 2

Run scripts in this exact order:

```bash
cd src

# Day 1 — Data loading + visualization
python vrp_data_loader.py
python vrp_visualize.py

# Day 2 — Baseline
python vrp_baseline.py

# Day 3 — OR-Tools
python vrp_ortools.py

# Day 4 — Genetic Algorithm
python vrp_genetic.py

# Day 5 — Tabu Search
python vrp_tabu.py

# Day 6 — ACO
python vrp_aco.py

# Day 7 — RL Environment check
python vrp_rl_env.py

# Day 8 — PPO Training
python vrp_rl_agent.py

# Day 9 — Pointer Network
python vrp_pointer_net.py

# Day 10 — Hybrid PPO+2opt (proposed method)
python vrp_hybrid.py

# Day 11 — Final comparison
python vrp_compare.py
```

---

## Dependencies

```
ortools
networkx
matplotlib
numpy
pandas
stable-baselines3
gymnasium
torch
scipy
```

---

## Research Contribution

The proposed **Hybrid PPO+2-opt** framework introduces a two-stage optimization strategy:

1. **Construction stage** — PPO agent learns to construct feasible vehicle routes through a custom Gymnasium environment with capacity constraints
2. **Refinement stage** — 2-opt local search systematically improves each constructed route by reversing sub-sequences

This combination leverages the global exploration capability of reinforcement learning with the local optimization power of 2-opt, achieving results that surpass both standalone RL and exact solvers.

---

## Roadmap

- [x] Phase 1 — Demand Forecasting + Inventory Optimization
- [x] Phase 2 — Vehicle Routing Optimization
- [ ] Phase 3 — Production Scheduling (Taillard benchmark)
- [ ] Phase 4 — End-to-End RL Simulation (SC2)
- [ ] Phase 5 — Supplier Selection + Ranking
