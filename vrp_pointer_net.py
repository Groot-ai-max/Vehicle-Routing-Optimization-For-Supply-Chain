import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from vrp_data_loader import load_cvrp

# ── Pointer Network Architecture ────────────────────────────────
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_ref   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_q     = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v       = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, query, ref, mask=None):
        ref_enc  = self.W_ref(ref)
        q_enc    = self.W_q(query).unsqueeze(1).expand_as(ref_enc)
        scores   = self.v(torch.tanh(ref_enc + q_enc)).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        return torch.softmax(scores, dim=-1)

class PointerNetwork(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, n_layers=1):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.encoder     = nn.LSTM(input_dim, hidden_dim,
                                   n_layers, batch_first=True)
        self.decoder_rnn = nn.LSTMCell(input_dim, hidden_dim)
        self.attention   = Attention(hidden_dim)
        self.input_proj  = nn.Linear(input_dim, hidden_dim)

    def forward(self, coords, mask=None):
        B, N, _  = coords.shape
        enc_out, (h, c) = self.encoder(coords)
        h = h.squeeze(0)
        c = c.squeeze(0)

        # Start from depot (index 0)
        decoder_input = coords[:, 0, :]
        tour          = []
        log_probs     = []
        visited       = torch.zeros(B, N, dtype=torch.bool,
                                    device=coords.device)
        visited = visited.clone()
        visited[:, 0] = True

        for step in range(N - 1):
            h, c     = self.decoder_rnn(decoder_input, (h, c))
            query    = h
            probs    = self.attention(query, enc_out, mask=visited)
            idx      = torch.argmax(probs, dim=1)
            tour.append(idx)
            log_probs.append(torch.log(probs.gather(1, idx.unsqueeze(1))
                                        .squeeze(1) + 1e-8))
            visited = visited.clone()
            visited[torch.arange(B), idx] = True
            decoder_input = coords[torch.arange(B), idx]

        tour      = torch.stack(tour, dim=1)
        log_probs = torch.stack(log_probs, dim=1).sum(dim=1)
        return tour, log_probs

# ── Tour length ──────────────────────────────────────────────────
def tour_length(coords, tour):
    B, N = tour.shape
    total = torch.zeros(B, device=coords.device)
    depot = coords[:, 0, :]

    for i in range(N):
        curr = coords[torch.arange(B), tour[:, i]]
        next_node = coords[torch.arange(B), tour[:, (i+1) % N]]
        total += torch.norm(curr - next_node, dim=1)

    # Add return to depot
    last = coords[torch.arange(B), tour[:, -1]]
    total += torch.norm(last - depot, dim=1)
    return total

# ── Generate random VRP instances for training ───────────────────
def generate_batch(batch_size, n_nodes, device):
    coords = torch.rand(batch_size, n_nodes, 2, device=device)
    return coords

# ── Train Pointer Network ────────────────────────────────────────
def train_pointer_network(n_nodes=20, batch_size=128,
                           n_epochs=20, lr=1e-3, device="cpu"):
    model     = PointerNetwork(input_dim=2, hidden_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history   = []

    print(f"  Training on {n_nodes}-node instances...")

    for epoch in range(1, n_epochs + 1):
        coords    = generate_batch(batch_size, n_nodes, device)
        tour, lp  = model(coords)
        length    = tour_length(coords, tour)
        baseline  = length.mean().detach()
        loss      = ((length - baseline) * (-lp)).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        mean_len = length.mean().item()
        history.append(mean_len)
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d} | "
                  f"Mean tour: {mean_len:.4f} | Loss: {loss.item():.4f}")

    return model, history

# ── Evaluate on real VRP instance ───────────────────────────────
def evaluate_on_instance(model, inst, device="cpu"):
    df     = inst["customers"]
    coords = df[["x", "y"]].values.astype(np.float32)

    # Normalize
    coords_norm = (coords - coords.min(0)) / (coords.max(0) - coords.min(0) + 1e-8)
    coords_t    = torch.tensor(coords_norm).unsqueeze(0).to(device)

    with torch.no_grad():
        tour, _ = model(coords_t)

    tour_np = tour.squeeze(0).cpu().numpy()

    # Calculate real distance
    dist_matrix = inst["dist_matrix"]
    depot       = inst["depot_idx"]
    total_dist  = 0
    prev        = depot

    for node in tour_np:
        total_dist += dist_matrix[prev][node]
        prev        = node
    total_dist += dist_matrix[prev][depot]

    return tour_np, round(total_dist, 2)

# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    CVRP_FOLDER = r"G:\RESEARCH\supply_chain_project\Transportation\XML"
    OUT_SCORES  = r"G:\RESEARCH\supply_chain_project\Transportation\results\model_scores"
    OUT_FIGS    = r"G:\RESEARCH\supply_chain_project\Transportation\results\figs"
    OUT_MODELS  = r"G:\RESEARCH\supply_chain_project\Transportation\results\models"
    os.makedirs(OUT_SCORES, exist_ok=True)
    os.makedirs(OUT_FIGS,   exist_ok=True)
    os.makedirs(OUT_MODELS, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print(f"Pointer Network — Device: {device}")
    print("=" * 60)

    # Train
    print("\nTraining Pointer Network...")
    t0 = time.time()
    model, history = train_pointer_network(
        n_nodes    = 20,
        batch_size = 128,
        n_epochs   = 50,
        lr         = 1e-3,
        device     = device
    )
    train_time = round(time.time() - t0, 2)

    # Save model
    torch.save(model.state_dict(),
               os.path.join(OUT_MODELS, "pointer_net.pth"))
    print(f"\n  Train time: {train_time}s")

    # Plot training curve
    plt.figure(figsize=(8, 4))
    plt.plot(history, color="purple", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Tour Length")
    plt.title("Pointer Network Training Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIGS, "pointer_net_training.png"), dpi=150)
    plt.close()

    # Evaluate on real instances
    print("\nEvaluating on real CVRP instances...")
    results    = []
    cvrp_files = sorted([f for f in os.listdir(CVRP_FOLDER)
                         if f.endswith(".vrp")])[:5]

    for fname in cvrp_files:
        inst = load_cvrp(os.path.join(CVRP_FOLDER, fname))
        t0   = time.time()
        tour, dist = evaluate_on_instance(model, inst, device)
        elapsed    = round(time.time() - t0, 3)
        print(f"  {fname:30s} | dist={dist:.2f} | time={elapsed}s")
        results.append({
            "instance"      : fname,
            "type"          : "CVRP",
            "method"        : "Pointer Network",
            "total_distance": dist,
            "num_vehicles"  : 1,
            "time_sec"      : elapsed
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_SCORES, "pointer_net_results.csv"), index=False)

    print("\n" + "=" * 60)
    print(df.to_string(index=False))
    print("\nSaved to results/model_scores/pointer_net_results.csv")
    print("Day 9 complete!")