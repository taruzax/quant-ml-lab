from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from lab.models.base import BaseModel
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
    ) -> None:
        super().__init__(task=task)
        if hidden_size <= 0 or num_layers <= 0 or epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
            raise ValueError("hidden_size, num_layers, epochs, batch_size and learning_rate must be positive")
        if not 0 <= output_dropout < 1 or not 0 <= recurrent_dropout < 1:
            raise ValueError("dropout values must be in [0, 1)")
        if num_layers == 1 and recurrent_dropout != 0:
            raise ValueError("recurrent_dropout requires at least two recurrent layers")
        if device != "cpu":
            raise ValueError("C15 supports only device='cpu'")
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.output_dropout = float(output_dropout)
        self.recurrent_dropout = float(recurrent_dropout)
        self.seed = int(seed)
        self.device = device
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
        if features.ndim != 3:
            raise ValueError("Recurrent models require features with shape [N, T, F]")
        self.feature_dim = int(features.shape[-1])
        network = self._ensure_network(self.feature_dim)
        if stopping_data is not None:
            stop_features, stop_targets = self.validate_inputs(*stopping_data)
            if stop_features.ndim != 3 or stop_features.shape[-1] != self.feature_dim:
                raise ValueError("Stopping features must have shape [N, T, F] matching training features")
            prepared_stopping = (stop_features.astype(np.float32), stop_targets)
        else:
            prepared_stopping = None
        self.training_history, self.stopping_history, selected = train_recurrent(
            network, features.astype(np.float32), targets, task=self.task,
            epochs=self.epochs, batch_size=self.batch_size, learning_rate=self.learning_rate,
            seed=self.seed, device=self.device, stopping_data=prepared_stopping,
            duration=duration, stopping_patience=stopping_patience,
        )
        self.selected_duration = selected
        self.stopping_config = {"patience": int(stopping_patience), "metric": "mse" if self.task == "regression" else "cross_entropy"} if stopping_data is not None else {}
        self.network.eval()
        return self

    def _features_tensor(self, X: Any) -> torch.Tensor:
        features = self.validate_features(X)
        if features.ndim != 3:
            raise ValueError("Recurrent models require features with shape [N, T, F]")
        if self.network is None or self.input_dim is None:
            raise RuntimeError("Recurrent model must be fitted before prediction")
        if features.shape[-1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, received {features.shape[-1]}")
        return torch.from_numpy(features.astype(np.float32))

    def predict(self, X: Any) -> np.ndarray:
        tensor = self._features_tensor(X)
        assert self.network is not None
        self.network.eval()
        with torch.no_grad():
            output = self.network(tensor).cpu().numpy()
        if self.task == "classification":
            return self.classes[np.argmax(output, axis=1)]
        return output.reshape(-1).astype(np.float64)

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba is available only for classification")
        tensor = self._features_tensor(X)
        assert self.network is not None
        self.network.eval()
        with torch.no_grad():
            return torch.softmax(self.network(tensor), dim=1).cpu().numpy()

    @classmethod
    def load(cls, directory: str) -> "_RecurrentModel":
        model = super().load(directory)
        assert isinstance(model, cls)
        if model.network is not None:
            model.network.to(torch.device("cpu"))
            model.network.eval()
        model.device = "cpu"
        return model


class GRUModel(_RecurrentModel):
    model_name = "gru"
    cell_type = "gru"


class LSTMModel(_RecurrentModel):
    model_name = "lstm"
    cell_type = "lstm"
