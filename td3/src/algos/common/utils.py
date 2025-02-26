import torch
import torch.nn as nn
from typing import Iterable

def mlp(
    sizes: tuple[int, ...],
    activation: type[nn.Module],
    output_activation: type[nn.Module],
):
    def build():
        for j in range(len(sizes) - 2):
            yield nn.Linear(sizes[j], sizes[j + 1])
            yield activation()

        # Final layer without non-linearity.
        yield nn.Linear(sizes[-2], sizes[-1])
        yield output_activation()

    return nn.Sequential(*build())

# Code for polyak update mostly from StableBaselines3
def polyak_update(
    params: Iterable[torch.Tensor],
    target_params: Iterable[torch.Tensor],
    tau: float,
) -> None:
    """
    Perform a Polyak average update on ``target_params`` using ``params``:
    target parameters are slowly updated towards the main parameters.
    ``tau``, the soft update coefficient controls the interpolation:
    ``tau=1`` corresponds to copying the parameters to the target ones whereas nothing happens when ``tau=0``.
    The Polyak update is done in place, with ``no_grad``, and therefore does not create intermediate tensors,
    or a computation graph, reducing memory cost and improving performance.  We scale the target params
    by ``1-tau`` (in-place), add the new weights, scaled by ``tau`` and store the result of the sum in the target
    params (in place).

    :param params: parameters to use to update the target params
    :param target_params: parameters to update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    """
    with torch.no_grad():
        for param, target_param in zip(params, target_params):
            target_param.data.mul_(1 - tau)
            torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)