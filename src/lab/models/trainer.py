from __future__ import annotations

import random
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from lab.core.contracts import WindowSet


def seed_torch(seed: int, deterministic_policy: str = "strict") -> None:
    """Seed Python, NumPy and PyTorch with an explicit deterministic policy."""
    if deterministic_policy not in {"strict", "warning"}:
        raise ValueError("deterministic_policy must be 'strict' or 'warning'")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=deterministic_policy == "warning")


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


class _WindowTargetDataset(Dataset):
    def __init__(self, windows: WindowSet, targets: torch.Tensor) -> None:
        self.windows = windows
        self.targets = targets

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[np.ndarray, torch.Tensor]:
        item = self.windows[index]
        features = item[0] if isinstance(item, tuple) else item
        return features, self.targets[index]


def _dataset(X: Any, y: np.ndarray, task: str) -> Dataset:
    targets = _targets(y, task)
    if len(targets) == 0:
        raise ValueError("Training targets must not be empty")
    if isinstance(X, WindowSet):
        if len(X) != len(targets):
            raise ValueError("Targets must align with window descriptors")
        return _WindowTargetDataset(X, targets)
    features = torch.from_numpy(np.asarray(X, dtype=np.float32))
    if features.ndim != 3 or features.shape[0] != len(targets):
        raise ValueError("Recurrent training data must align with [N, T, F] features")
    return TensorDataset(features, targets)


def _loss(model: nn.Module, dataset: Dataset, criterion: nn.Module, *, batch_size: int, device: torch.device) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    weighted_loss = 0.0
    sample_count = 0
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_loss = criterion(model(batch_X), batch_y)
            batch_count = len(batch_X)
            weighted_loss += float(batch_loss.item()) * batch_count
            sample_count += batch_count
    return weighted_loss / sample_count


def train_recurrent(
    model: nn.Module,
    X: Any,
    y: np.ndarray,
    *,
    task: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
    stopping_data: tuple[Any, np.ndarray] | None = None,
    duration: int | None = None,
    stopping_patience: int = 5,
    deterministic_policy: str = "strict",
) -> tuple[list[dict[str, float]], list[dict[str, float]] | None, int]:
    """Train a recurrent network and optionally select duration on a stopping set."""
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")
    if duration is not None and duration <= 0:
        raise ValueError("duration must be positive")
    if stopping_data is not None and stopping_patience <= 0:
        raise ValueError("stopping_patience must be positive")
    seed_torch(seed, deterministic_policy)
    torch_device = torch.device(device)
    model.to(torch_device)
    dataset = _dataset(X, y, task)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    criterion = _loss_function(task)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    max_epochs = epochs if duration is None else duration
    stopping_dataset = None
    if stopping_data is not None:
        stopping_dataset = _dataset(stopping_data[0], stopping_data[1], task)
    training_history: list[dict[str, float]] = []
    stopping_history: list[dict[str, float]] | None = [] if stopping_data is not None else None
    best_state: dict[str, Any] | None = None
    best_loss = float("inf")
    best_epoch = max_epochs
    stale_epochs = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        weighted_loss = 0.0
        sample_count = 0
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(torch_device)
            batch_y = batch_y.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            batch_count = len(batch_X)
            weighted_loss += float(loss.detach().cpu().item()) * batch_count
            sample_count += batch_count
        train_loss = weighted_loss / sample_count
        training_history.append({"epoch": float(epoch), "loss": train_loss})
        if stopping_dataset is None:
            continue
        stop_loss = _loss(model, stopping_dataset, criterion, batch_size=batch_size, device=torch_device)
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
