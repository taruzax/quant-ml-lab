from __future__ import annotations

import random
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def seed_torch(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for deterministic CPU training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _targets(y: np.ndarray, task: str) -> torch.Tensor:
    values = np.asarray(y).reshape(-1)
    if task == "classification":
        if not np.isfinite(values).all() or not np.equal(values, values.astype(int)).all():
            raise ValueError("Classification targets must be finite integer labels")
        labels = values.astype(np.int64)
        if not np.isin(labels, (-1, 0, 1)).all():
            raise ValueError("Classification labels must be one of -1, 0, +1")
        return torch.from_numpy(labels + 1)
    return torch.from_numpy(values.astype(np.float32)).reshape(-1, 1)


def _loss_function(task: str) -> nn.Module:
    return nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()


def _loss(model: nn.Module, X: torch.Tensor, y: torch.Tensor, criterion: nn.Module) -> float:
    model.eval()
    with torch.no_grad():
        output = model(X)
        value = criterion(output, y if y.ndim == 1 else y).item()
    return float(value)


def train_recurrent(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
    stopping_data: tuple[np.ndarray, np.ndarray] | None = None,
    duration: int | None = None,
    stopping_patience: int = 5,
) -> tuple[list[dict[str, float]], list[dict[str, float]] | None, int]:
    """Train a recurrent network and optionally select duration on a stopping set."""
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")
    if duration is not None and duration <= 0:
        raise ValueError("duration must be positive")
    if stopping_data is not None and stopping_patience <= 0:
        raise ValueError("stopping_patience must be positive")
    seed_torch(seed)
    torch_device = torch.device(device)
    model.to(torch_device)
    train_X = torch.from_numpy(np.asarray(X, dtype=np.float32))
    train_y = _targets(y, task)
    dataset = TensorDataset(train_X, train_y)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    criterion = _loss_function(task)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    max_epochs = epochs if duration is None else duration
    stop_X = stop_y = None
    if stopping_data is not None:
        stop_X = torch.from_numpy(np.asarray(stopping_data[0], dtype=np.float32)).to(torch_device)
        stop_y = _targets(stopping_data[1], task).to(torch_device)
    training_history: list[dict[str, float]] = []
    stopping_history: list[dict[str, float]] | None = [] if stopping_data is not None else None
    best_state: dict[str, Any] | None = None
    best_loss = float("inf")
    best_epoch = max_epochs
    stale_epochs = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses: list[float] = []
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(torch_device)
            batch_y = batch_y.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        train_loss = float(np.mean(losses))
        training_history.append({"epoch": float(epoch), "loss": train_loss})
        if stop_X is None or stop_y is None:
            continue
        stop_loss = _loss(model, stop_X, stop_y, criterion)
        assert stopping_history is not None
        stopping_history.append({"epoch": float(epoch), "loss": stop_loss})
        if stop_loss < best_loss:
            best_loss = stop_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= stopping_patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return training_history, stopping_history, int(best_epoch)
