from collections.abc import Sequence

import torch as th

_MASK_32 = 0xFFFFFFFF
_GOLDEN_RATIO_32 = 0x9E3779B9


def rand_uniform(
    num: int,
    num_draws: int,
    device: th.device | str,
    seeds: Sequence[int] | None = None,
) -> th.Tensor:
    """
    Returns [num, num_draws] float32 values uniformly distributed in [0, 1) on the `device`.
    Unseeded, it samples from the global torch RNG. With the `seeds` (one per row, `num` of them),
    the values are reproducible and identical on any device (see `hash_uniform()`).
    """
    if seeds is None:
        return th.rand(num, num_draws, device=device)
    if len(seeds) != num:
        raise ValueError(f"Got {len(seeds)} seeds for {num} values.")
    return hash_uniform(seeds=seeds, num_draws=num_draws, device=device)


def hash_uniform(
    seeds: Sequence[int], num_draws: int, device: th.device | str
) -> th.Tensor:
    """
    Returns [len(seeds), num_draws] float32 values in [0, 1), the `j`-th draw of a seed being
    a hash of the (seed, j) pair. The hash uses only exact integer operations, so the values
    are bit-identical on the CPU and the GPU, and don't depend on the torch RNG implementation
    (unlike `th.Generator`, which gives different values for the same seed on each device).
    """
    seeds_t = th.as_tensor(list(seeds), dtype=th.int64, device=device)
    if seeds_t.numel() and (seeds_t.min() < 0 or seeds_t.max() > _MASK_32):
        raise ValueError(f"The seeds must be in [0, 2^32), got {list(seeds)}.")
    draws = th.arange(1, num_draws + 1, dtype=th.int64, device=device)
    # Weyl sequence step per draw, mixed with the hashed seed
    keys = (_lowbias32(seeds_t)[:, None] + draws[None, :] * _GOLDEN_RATIO_32) & _MASK_32
    # The top 24 bits are exactly representable as float32
    return (_lowbias32(keys) >> 8).to(th.float32) / float(1 << 24)


def _lowbias32(x: th.Tensor) -> th.Tensor:
    """The `lowbias32` integer hash (C. Wellons) of the 32-bit values in the int64 `x`."""
    x = x ^ (x >> 16)
    x = _mul_mod_32(x, 0x7FEB352D)
    x = x ^ (x >> 15)
    x = _mul_mod_32(x, 0x846CA68B)
    return x ^ (x >> 16)


def _mul_mod_32(x: th.Tensor, c: int) -> th.Tensor:
    """(x * c) mod 2^32 for the 32-bit `x` and `c`, without the int64 overflow."""
    lo = x & 0xFFFF
    hi = x >> 16
    return (lo * c + (((hi * c) & 0xFFFF) << 16)) & _MASK_32
