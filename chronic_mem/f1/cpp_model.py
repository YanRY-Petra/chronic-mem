"""
Optional Clinical Persistence Predictor (CPP) for dynamic λ_i estimation in F1.

PyTorch is imported lazily when `build_cpp_model` / `ClinicalPersistencePredictor`
is instantiated, keeping rule-only cold-start mode lightweight (< 30 MB RAM).
"""
from __future__ import annotations

from typing import Any


class ClinicalPersistencePredictor:
    """
    3-Layer MLP predicting bounded retention horizon λ_i in [min_days, max_days]:
        λ_i = min_days + (max_days - min_days) * softplus(raw) / (1 + softplus(raw))
    Input: v_umls (16-d) ⊕ v_context (128-d) ⊕ φ (1-d).
    """

    def __init__(
        self,
        umls_dim: int = 16,
        context_dim: int = 128,
        hidden_dim: int = 64,
        min_days: int = 1,
        max_days: int = 10000,
    ):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        self._torch = torch
        self._F = F
        self.umls_dim = umls_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.min_days = min_days
        self.max_days = max_days

        input_dim = umls_dim + context_dim + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def parameters(self):
        return self.net.parameters()

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, state_dict: Any):
        return self.net.load_state_dict(state_dict)

    def eval(self):
        self.net.eval()
        return self

    def forward(self, v_umls: Any, v_context: Any, phi: Any) -> Any:
        torch = self._torch
        F = self._F
        x = torch.cat([v_umls, v_context, phi], dim=1)
        raw = self.net(x)
        s = F.softplus(raw)
        w = s / (1.0 + s)
        return self.min_days + (self.max_days - self.min_days) * w

    def predict_lambda(self, v_umls: Any, v_context: Any, phi: float) -> float:
        torch = self._torch
        self.net.eval()
        with torch.no_grad():
            u = torch.as_tensor(v_umls, dtype=torch.float32).reshape(1, -1)
            c = torch.as_tensor(v_context, dtype=torch.float32).reshape(1, -1)
            p = torch.tensor([[float(phi)]], dtype=torch.float32)
            return float(self.forward(u, c, p).item())
