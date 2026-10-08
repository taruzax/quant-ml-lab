from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lab.core.contracts import WindowSet
from lab.models.base import BaseModel
from lab.models.devices import resolve_device
from lab.models.trainer import train_recurrent


class _RecurrentNetwork(nn.Module):
    def __init__(
        self,
        cell_type: str,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_dropout: float,
        recurrent_dropout: float,
        task: str,
    ) -> None:
        super().__init__()
        recurrent_cls = nn.GRU if cell_type == "gru" else nn.LSTM
        self.recurrent = recurrent_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout if num_layers > 1 else 0.0,
        )
        self.output_dropout = nn.Dropout(output_dropout)
        self.head = nn.Linear(hidden_size, 3 if task == "classification" else 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.recurrent(X)
        return self.head(self.output_dropout(sequence[:, -1, :]))


class _RecurrentModel(BaseModel):
    cell_type = "recurrent"

    def __init__(
        self,
        *,
        task: str = "regression",
        hidden_size: int = 32,
        num_layers: int = 1,
        learning_rate: float = 0.001,
        epochs: int = 20,
        batch_size: int = 32,
        output_dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        seed: int = 42,
        device: str = "cpu",
        deterministic_policy: str = "strict",
    ) -> None:
        super().__init__(task=task)
        if hidden_size <= 0 or num_layers <= 0 or epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
            raise ValueError("hidden_size, num_layers, epochs, batch_size and learning_rate must be positive")
        if not 0 <= output_dropout < 1 or not 0 <= recurrent_dropout < 1:
            raise ValueError("dropout values must be in [0, 1)")
        if num_layers == 1 and recurrent_dropout != 0:
            raise ValueError("recurrent_dropout requires at least two recurrent layers")
        resolved_device = resolve_device(self.model_name, device).resolved_device
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.output_dropout = float(output_dropout)
        self.recurrent_dropout = float(recurrent_dropout)
        self.seed = int(seed)
        self.device = resolved_device
        self.deterministic_policy = deterministic_policy
        self.input_dim: int | None = None
        self.network: _RecurrentNetwork | None = None
        self.classes = np.asarray([-1, 0, 1], dtype=np.int8)

    def _ensure_network(self, input_dim: int) -> _RecurrentNetwork:
        if self.network is None:
            self.input_dim = int(input_dim)
            self.network = _RecurrentNetwork(
                self.cell_type, input_dim, self.hidden_size, self.num_layers,
                self.output_dropout, self.recurrent_dropout, self.task,
            )
        elif self.input_dim != input_dim:
            raise ValueError(f"Expected {self.input_dim} features, received {input_dim}")
        return self.network

    def fit(
        self,
        X: Any,
        y: Any,
        stopping_data: tuple[Any, Any] | None = None,
        *,
        duration: int | None = None,
        stopping_patience: int = 5,
    ) -> "_RecurrentModel":
        features, targets = self.validate_inputs(X, y)
        assert targets is not None
        if len(features.shape) != 3:
            raise ValueError("Recurrent models require features with shape [N, T, F]")
        self.feature_dim = int(features.shape[-1])
        network = self._ensure_network(self.feature_dim)
        if stopping_data is not None:
            stop_features, stop_targets = self.validate_inputs(*stopping_data)
            if len(stop_features.shape) != 3 or stop_features.shape[-1] != self.feature_dim:
                raise ValueError("Stopping features must have shape [N, T, F] matching training features")
            assert stop_targets is not None
            prepared_stopping = (stop_features, stop_targets)
        else:
            prepared_stopping = None
        self.training_history, self.stopping_history, selected = train_recurrent(
            network, features, targets, task=self.task,
            epochs=self.epochs, batch_size=self.batch_size, learning_rate=self.learning_rate,
            seed=self.seed, device=self.device, stopping_data=prepared_stopping,
            duration=duration, stopping_patience=stopping_patience,
            deterministic_policy=self.deterministic_policy,
        )
        self.selected_duration = selected
        self.stopping_config = {"patience": int(stopping_patience), "metric": "mse" if self.task == "regression" else "cross_entropy"} if stopping_data is not None else {}
        self.network.eval()
        return self

    def _features_loader(self, X: Any) -> DataLoader:
        features = self.validate_features(X)
        if len(features.shape) != 3:
            raise ValueError("Recurrent models require features with shape [N, T, F]")
        if self.network is None or self.input_dim is None:
            raise RuntimeError("Recurrent model must be fitted before prediction")
        if features.shape[-1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, received {features.shape[-1]}")
        if isinstance(features, WindowSet):
            dataset = features
        else:
            tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))
            dataset = TensorDataset(tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=0)

    def predict(self, X: Any) -> np.ndarray:
        loader = self._features_loader(X)
        assert self.network is not None
        self.network.eval()
        batches = []
        with torch.no_grad():
            for batch in loader:
                features = batch[0] if isinstance(batch, (tuple, list)) else batch
                batches.append(self.network(features.to(torch.device(self.device))).cpu().numpy())
        output = np.concatenate(batches, axis=0)
        if self.task == "classification":
            return self.classes[np.argmax(output, axis=1)]
        return output.reshape(-1).astype(np.float64)

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba is available only for classification")
        loader = self._features_loader(X)
        assert self.network is not None
        self.network.eval()
        batches = []
        with torch.no_grad():
            for batch in loader:
                features = batch[0] if isinstance(batch, (tuple, list)) else batch
                batches.append(torch.softmax(
                    self.network(features.to(torch.device(self.device))), dim=1
                ).cpu().numpy())
        return np.concatenate(batches, axis=0)

    @classmethod
    def load(cls, directory: str, device: str | None = None) -> "_RecurrentModel":
        model = super().load(directory)
        assert isinstance(model, cls)
        if model.network is not None:
            resolved = resolve_device(model.model_name, device or "cpu").resolved_device
            model.network.to(torch.device(resolved))
            model.network.eval()
            model.device = resolved
        return model


class GRUModel(_RecurrentModel):
    model_name = "gru"
    cell_type = "gru"


class LSTMModel(_RecurrentModel):
    model_name = "lstm"
    cell_type = "lstm"
