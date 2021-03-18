import torch
import numpy as np
from torch import nn
from typing import Any, Dict, Tuple, Union, Optional, Sequence


class CURL_DQN(nn.Module):
    """Reference: Human-level control through deep reinforcement learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        c: int,
        h: int,
        w: int,
        action_shape: Sequence[int],
        device: Union[str, int, torch.device] = "cpu",
        features_only: bool = False,
    ) -> None:
        super().__init__()
        self.device = device
        self.convs = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
            nn.Flatten())
        with torch.no_grad():
            self.output_dim = np.prod(
                self.convs(torch.zeros(1, c, h, w)).shape[1:])
        if not features_only:
            self.fc = nn.Sequential(
                nn.Linear(self.output_dim, 512), nn.ReLU(inplace=True),
                nn.Linear(512, np.prod(action_shape)))
            self.contrastive_head = nn.Sequential(
                nn.Linear(self.output_dim, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.LayerNorm(128)
            )
            self.W = nn.Parameter(torch.rand(128, 128))
            self.output_dim = np.prod(action_shape)

    def forward(
        self,
        x: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Q(x, \*)."""
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        h = self.convs(x)

        h_ = self.contrastive_head(h)
        q = self.fc(h)
        return q, h_, state


# class C51(DQN):
#     """Reference: A distributional perspective on reinforcement learning.

#     For advanced usage (how to customize the network), please refer to
#     :ref:`build_the_network`.
#     """

#     def __init__(
#         self,
#         c: int,
#         h: int,
#         w: int,
#         action_shape: Sequence[int],
#         num_atoms: int = 51,
#         device: Union[str, int, torch.device] = "cpu",
#     ) -> None:
#         self.action_num = np.prod(action_shape)
#         super().__init__(c, h, w, [self.action_num * num_atoms], device)
#         self.num_atoms = num_atoms

#     def forward(
#         self,
#         x: Union[np.ndarray, torch.Tensor],
#         state: Optional[Any] = None,
#         info: Dict[str, Any] = {},
#     ) -> Tuple[torch.Tensor, Any]:
#         r"""Mapping: x -> Z(x, \*)."""
#         x, state = super().forward(x)
#         x = x.view(-1, self.num_atoms).softmax(dim=-1)
#         x = x.view(-1, self.action_num, self.num_atoms)
#         return x, state


# class QRDQN(DQN):
#     """Reference: Distributional Reinforcement Learning with Quantile \
#     Regression.

#     For advanced usage (how to customize the network), please refer to
#     :ref:`build_the_network`.
#     """

#     def __init__(
#         self,
#         c: int,
#         h: int,
#         w: int,
#         action_shape: Sequence[int],
#         num_quantiles: int = 200,
#         device: Union[str, int, torch.device] = "cpu",
#     ) -> None:
#         self.action_num = np.prod(action_shape)
#         super().__init__(c, h, w, [self.action_num * num_quantiles], device)
#         self.num_quantiles = num_quantiles

#     def forward(
#         self,
#         x: Union[np.ndarray, torch.Tensor],
#         state: Optional[Any] = None,
#         info: Dict[str, Any] = {},
#     ) -> Tuple[torch.Tensor, Any]:
#         r"""Mapping: x -> Z(x, \*)."""
#         x, state = super().forward(x)
#         x = x.view(-1, self.action_num, self.num_quantiles)
#         return x, state
