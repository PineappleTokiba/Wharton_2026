
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from torch import nn
from src.preprocessing import Snapshot

class GraphConvolution(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return adjacency @ self.linear(features)

class StockGNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 16) -> None:
        super().__init__()
        self.conv1 = GraphConvolution(input_size, hidden_size)
        self.conv2 = GraphConvolution(hidden_size, 1)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.conv1(features, adjacency))
        return self.conv2(hidden, adjacency).squeeze(-1)

class NodeMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 16) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        del adjacency
        return self.layers(features).squeeze(-1)

def standardization_parameters(snapshots: Sequence[Snapshot]) -> Tuple[np.ndarray, np.ndarray]:
    values = np.concatenate([snapshot.features for snapshot in snapshots], axis=0)
    mean = values.mean(axis=0)
    standard_deviation = values.std(axis=0)
    standard_deviation[standard_deviation < 1e-8] = 1.0
    return mean.astype(np.float32), standard_deviation.astype(np.float32)

def normalize_features(features: np.ndarray, mean: np.ndarray, standard_deviation: np.ndarray) -> np.ndarray:
    return ((features - mean) / standard_deviation).astype(np.float32)

def _tensor_snapshot(snapshot: Snapshot, mean: np.ndarray, standard_deviation: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = normalize_features(snapshot.features, mean, standard_deviation)
    return (torch.from_numpy(features), torch.from_numpy(snapshot.adjacency), torch.from_numpy(snapshot.labels))

def _evaluate_loss(model: nn.Module, snapshots: Sequence[Snapshot], mean: np.ndarray, standard_deviation: np.ndarray) -> float:
    model.eval()
    loss_function = nn.BCEWithLogitsLoss()
    losses: List[float] = []
    with torch.no_grad():
        for snapshot in snapshots:
            features, adjacency, labels = _tensor_snapshot(snapshot, mean, standard_deviation)
            logits = model(features, adjacency)
            losses.append(float(loss_function(logits, labels).item()))
    return float(np.mean(losses))

def train_model(
    model: nn.Module, train: Sequence[Snapshot], validation: Sequence[Snapshot],
    mean: np.ndarray, standard_deviation: np.ndarray,
    max_epochs: int = 100, patience: int = 12, learning_rate: float = 0.01
) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_without_improvement = 0

    for _ in range(max_epochs):
        model.train()
        for snapshot in train:
            features, adjacency, labels = _tensor_snapshot(snapshot, mean, standard_deviation)
            optimizer.zero_grad()
            loss = loss_function(model(features, adjacency), labels)
            loss.backward()
            optimizer.step()

        validation_loss = _evaluate_loss(model, validation, mean, standard_deviation)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint")
    model.load_state_dict(best_state)
    return model