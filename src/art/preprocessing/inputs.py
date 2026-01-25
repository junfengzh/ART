from typing import TYPE_CHECKING

import torch

from .pack import PackedTensors

if TYPE_CHECKING:
    from .. import dev, types


class TrainInputs(PackedTensors):
    """Training inputs with config attached."""

    config: "types.TrainConfig"
    _config: "dev.TrainConfig"
    return_new_logprobs: bool


def create_train_inputs(
    packed_tensors: PackedTensors,
    offset: int,
    config: "types.TrainConfig",
    _config: "dev.TrainConfig",
    warmup: bool,
    batch_size: int = 1,
) -> TrainInputs:
    """Create TrainInputs for a batch of sequences.
    
    Args:
        packed_tensors: Packed tensor data
        offset: Starting offset in the packed tensors
        config: Training configuration
        _config: Development configuration
        warmup: Whether this is a warmup batch
        batch_size: Number of sequences to include in the batch
    """
    end_offset = min(offset + batch_size, packed_tensors["tokens"].shape[0])
    return TrainInputs(
        **{
            k: (
                v[offset:end_offset, :1024]
                if warmup and v.dim() > 1
                else v[offset:end_offset]
            )
            for k, v in packed_tensors.items()
            if isinstance(v, torch.Tensor)
        },
        pixel_values=(
            [None] * (end_offset - offset) if warmup else packed_tensors["pixel_values"][offset:end_offset]
        ),
        image_grid_thw=(
            [None] * (end_offset - offset) if warmup else packed_tensors["image_grid_thw"][offset:end_offset]
        ),
        config=(
            config.model_copy(update={"lr": 1e-9, "beta": 0.0, "kl_coef": 0.0})
            if warmup
            else config
        ),
        _config=_config,
        return_new_logprobs=False,
    )
