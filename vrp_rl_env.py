import numpy as np
import gymnasium as gym
from gymnasium import spaces

class VRPEnv(gym.Env):
    """
    Custom Gymnasium environment for CVRP.
    State  : [current_node_x, current_node_y, remaining_capacity,
               visited_mask (n_nodes), depot_x, depot_y]
    Action : next customer to visit (0 = return to depot)
    Reward : negative distance traveled
    """

    def __init__(self, inst, max_steps=None):
        super().__init__()
        self.inst        = inst
        self.dist        = inst["dist_matrix"]
        self.demands     = inst["customers"]["demand"].values
        self.capacity    = inst["capacity"]
        self.depot_idx   = inst["depot_idx"]
        self.n_nodes     = inst["n_nodes"]
        self.coords      = inst["customers"][["x", "y"]].values
        self.max_steps   = max_steps or self.n_nodes * 3

        # Normalize coords
        max_coord        = self.coords.max()
        self.coords_norm = self.coords / max_coord

        # Action space: which node to visit next
        self.action_space = spaces.Discrete(self.n_nodes)

        # Observation space:
        # [current_x, current_y, capacity_ratio, visited_mask...]
        obs_size = 3 + self.n_nodes
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(obs_size,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_node      = self.depot_idx
        self.remaining_capacity = self.capacity
        self.visited           = np.zeros(self.n_nodes, dtype=np.float32)
        self.visited[self.depot_idx] = 1.0
        self.total_distance    = 0.0
        self.steps             = 0
        self.routes            = [[self.depot_idx]]
        self.current_route     = 0
        return self._get_obs(), {}

    def _get_obs(self):
        cx, cy   = self.coords_norm[self.current_node]
        cap_ratio = self.remaining_capacity / self.capacity
        return np.concatenate([
            [cx, cy, cap_ratio],
            self.visited
        ]).astype(np.float32)

    def _get_valid_actions(self):
        valid = []
        for i in range(self.n_nodes):
            if i == self.depot_idx:
                continue
            if self.visited[i] == 0:
                if self.demands[i] <= self.remaining_capacity:
                    valid.append(i)
        return valid

    def step(self, action):
        self.steps  += 1
        valid        = self._get_valid_actions()
        all_visited  = len(valid) == 0

        # Force return to depot if no valid actions
        if all_visited or (action == self.depot_idx and len(valid) > 0):
            action = self.depot_idx

        # Invalid action — penalize
        if action not in valid and action != self.depot_idx:
            return self._get_obs(), -1.0, False, False, {}

        # Calculate reward
        dist    = self.dist[self.current_node][action]
        reward  = -dist / 1000.0   # normalize reward

        self.total_distance    += dist
        self.current_node       = action

        if action == self.depot_idx:
            # Starting new route
            self.remaining_capacity = self.capacity
            self.routes.append([self.depot_idx])
            self.current_route += 1
        else:
            self.visited[action]     = 1.0
            self.remaining_capacity -= self.demands[action]
            self.routes[self.current_route].append(action)

        # Check termination
        all_visited = all(
            self.visited[i] == 1.0
            for i in range(self.n_nodes)
            if i != self.depot_idx
        )

        if all_visited:
            # Return to depot
            dist = self.dist[self.current_node][self.depot_idx]
            self.total_distance += dist
            reward -= dist / 1000.0
            for route in self.routes:
                if route[-1] != self.depot_idx:
                    route.append(self.depot_idx)
            return self._get_obs(), reward, True, False, {}

        terminated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated, False, {}

    def get_solution(self):
        routes = [r for r in self.routes if len(r) > 2]
        return routes, round(self.total_distance, 2)


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from vrp_data_loader import load_cvrp

    CVRP_FILE = r"G:\RESEARCH\supply_chain_project\Transportation\XML\XML100_1111_01.vrp"
    inst      = load_cvrp(CVRP_FILE)

    print("=" * 55)
    print("VRP Gym Environment — Sanity Check")
    print("=" * 55)

    env = VRPEnv(inst)
    obs, _ = env.reset()

    print(f"Observation shape : {obs.shape}")
    print(f"Action space      : {env.action_space}")
    print(f"Obs space         : {env.observation_space}")
    print(f"N nodes           : {env.n_nodes}")
    print(f"Capacity          : {env.capacity}")

    # Random policy test
    total_reward = 0
    done         = False
    steps        = 0

    while not done and steps < 500:
        valid = env._get_valid_actions()
        action = valid[0] if valid else env.depot_idx
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done          = terminated or truncated
        steps        += 1

    routes, dist = env.get_solution()
    print(f"\nRandom policy test:")
    print(f"  Steps        : {steps}")
    print(f"  Total reward : {total_reward:.4f}")
    print(f"  Routes       : {len(routes)}")
    print(f"  Distance     : {dist:.2f}")
    print("\nEnvironment ready for RL training!")
    print("=" * 55)