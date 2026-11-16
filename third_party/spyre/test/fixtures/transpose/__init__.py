"""Transpose and permute copies of stick-tiled tensors.

One whole tensor per launch, no distribution loop: pure data-movement copies,
plus the same transposes fused into an add.
See ``meta.py`` for the cases and why each is there.
"""
