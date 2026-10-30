"""Transpose and permute copies of torch-spyre stick-tiled tensors.

One whole tensor per launch, no distribution loop: the pure data-movement
copies the scheduler re-sticks on chip (its DataTransferAlignment pass).
See ``meta.py`` for the cases and why each is there.
"""
