"""SIGNATURE + VARIANTS + reference oracle + input generator for transpose.

``y = x.permute(perm)`` of a torch-spyre fp16 tensor in its device layout, one
whole tensor per launch (``grid = [1]``, no distribution loop -- dbo-opt
rejects the ``scf.for`` a looped kernel outlines from its program id).

What the cases exercise. On the device every such copy is a *re-stick*: the
input's contiguous (stick) dim is not the output's, so the scheduler's
DataTransferAlignment pass reads whole 64x64 blocks and transposes them on
chip. The ``spyrecode`` stage has to hand it the copy in the one form it
accepts -- an input view whose strides walk the output's physical index
space, an identity ``linalg.generic``, a row-major output view -- which
StridedCopyFromDataMovementGeneric produces from the physicalized transpose.
That stage runs only for a ``compiles_to_binary`` variant on the device
(``test_device_launch.py``). The ``ktir_cpu`` tier runs the ``ktir`` artifact,
which is logical (a ``linalg.transpose`` between row-major views), so here it
checks only that the kernel lowers and permutes correctly before
physicalization; the re-stick form itself is pinned by
``test/Dialect/KTDP/Transforms/StridedCopyFromDataMovementGeneric/`` and
``test/spyre-triton-opt/stage-pipelines-restick.mlir``.

Shapes. Every dim is a multiple of the stick (64 fp16 elements), because
DataTransferAlignment transposes square stick-by-stick blocks; the rank-2
shapes include both non-square orientations and several blocks along each
dim. The rank-3 rows are one transpose of the two inner dims, one of the outer
two (the stick dim moves to dim 0), and both 3-cycles.

Values are integers in fp16's exact range, and a copy is exact, so the
tolerance is zero.

See ``fixtures/README.md`` for the field reference and discovery rules.
"""

from dataclasses import dataclass

import numpy as np

import conftest
from . import kernel
from utils import sticksize

SIGNATURE_2D = {
    "x_ptr": "*fp16",
    "y_ptr": "*fp16",
    "M": "constexpr",
    "N": "constexpr",
    "X_LAYOUT": "constexpr",
    "Y_LAYOUT": "constexpr",
}

SIGNATURE_3D = {
    "x_ptr": "*fp16",
    "y_ptr": "*fp16",
    **{name: "constexpr" for name in
       ("A", "B", "C", "P0", "P1", "P2", "OA", "OB", "OC", "X_LAYOUT", "Y_LAYOUT")},
}

#: The module-level signature the loader requires; ``permute3d`` overrides it.
SIGNATURE = SIGNATURE_2D

#: Elements in one stick of the fp16 tensors these kernels move.
_STICK = sticksize(SIGNATURE_2D, "x_ptr")

#: Largest magnitude of an input value; fp16 represents every integer up to
#: 2048 exactly, so the integers in [-_VALUE_RANGE, _VALUE_RANGE] survive the
#: round trip unchanged.
_VALUE_RANGE = 1000


def stick_layout(rank: int) -> tuple:
    """torch-spyre's device layout of a rank-*rank* tensor, labelled, as
    ``tl.spyre_tensor_layout`` takes it.

    The last logical dim is cut into sticks; the stick index sits where the
    rule below puts it and the first logical dim moves just outside the lane:
    rank 2 ``[R, C] -> [C // S, R, C % S]``, rank 3
    ``[D0, D1, D2] -> [D1, D2 // S, D0, D2 % S]``.
    """
    last = rank - 1
    middle = tuple(range(1, last))
    return ("stick", middle + ((last, "floordiv", _STICK), 0, (last, "mod", _STICK)))


def permute_row(shape: tuple, perm: tuple) -> tuple:
    """One ``(A, B, C, P0, P1, P2, OA, OB, OC)`` row: *shape* permuted by
    *perm*, with the output shape derived so a row cannot disagree with it."""
    return (*shape, *perm, *(shape[p] for p in perm))


@dataclass(frozen=True)
class Permute(conftest.VariantFactory):
    """Inputs and oracle of a permute of the given rank.

    The shape and permutation come from the variant's swept params:
    ``(M, N)`` and the fixed permutation ``(1, 0)`` at rank 2,
    ``(A, B, C)`` and ``(P0, P1, P2)`` at rank 3.
    """

    rank: int

    def _shape_and_perm(self, combo: dict) -> tuple:
        if self.rank == 2:
            return (combo["M"], combo["N"]), (1, 0)
        return ((combo["A"], combo["B"], combo["C"]),
                (combo["P0"], combo["P1"], combo["P2"]))

    def inputs(self, **combo):
        shape, perm = self._shape_and_perm(combo)

        def make(**_):
            rng = np.random.default_rng(0)
            x = rng.integers(-_VALUE_RANGE, _VALUE_RANGE + 1, size=shape)
            out_shape = tuple(shape[p] for p in perm)
            return {"x_ptr": x.astype(np.float16),
                    "y_ptr": np.zeros(out_shape, dtype=np.float16)}
        return make

    def reference(self, **combo):
        _, perm = self._shape_and_perm(combo)
        return lambda inputs: np.ascontiguousarray(
            np.transpose(inputs["x_ptr"], perm))


VARIANTS = {
    "default": {
        "summary": "y = x.T of a [M, N] fp16 stick-tiled tensor, one tile.",
        "kernel_fn": kernel.trans2d_stick,
        "factory": Permute(rank=2),
        "constexpr": ["M", "N", "X_LAYOUT", "Y_LAYOUT"],
        "params": {
            # Square, both non-square orientations, and several blocks along
            # each dim in both orientations.
            ("M", "N"): [(64, 64), (128, 64), (64, 128), (128, 256), (256, 128)],
            "X_LAYOUT": [stick_layout(2)],
            "Y_LAYOUT": [stick_layout(2)],
        },
        "grid": [1],
        "compiles_to_binary": True,
        "output_key": "y_ptr",
        "rtol": 0.0,
        "atol": 0.0,
        "tags": ["descriptor-load-static", "descriptor-store-static",
                 "simplified:no-loop", "spyre-tensor-layout", "restick"],
    },
    "permute3d": {
        "base": None,
        "summary": "y = x.permute(P0, P1, P2) of a rank-3 fp16 stick-tiled tensor.",
        "kernel_fn": kernel.permute3d_stick,
        "SIGNATURE": SIGNATURE_3D,
        "factory": Permute(rank=3),
        "constexpr": ["A", "B", "C", "P0", "P1", "P2", "OA", "OB", "OC",
                      "X_LAYOUT", "Y_LAYOUT"],
        "params": {
            ("A", "B", "C", "P0", "P1", "P2", "OA", "OB", "OC"): [
                permute_row((128, 64, 128), (0, 2, 1)),   # transpose(1, 2)
                permute_row((128, 128, 128), (2, 1, 0)),  # transpose(0, 2)
                permute_row((128, 64, 128), (2, 0, 1)),   # 3-cycle
                permute_row((128, 64, 128), (1, 2, 0)),   # the other 3-cycle
            ],
            "X_LAYOUT": [stick_layout(3)],
            "Y_LAYOUT": [stick_layout(3)],
        },
        "grid": [1],
        "compiles_to_binary": True,
        "output_key": "y_ptr",
        "rtol": 0.0,
        "atol": 0.0,
        "tags": ["descriptor-load-static", "descriptor-store-static",
                 "simplified:no-loop", "spyre-tensor-layout", "restick"],
        # ktir_cpu charges a whole loaded tile to the core's 2 MiB LX
        # scratchpad, and each of these tensors is 2 MiB of fp16. The device
        # streams the copy through 64x64 staging blocks instead, so only the
        # interpreter tier is affected; permute3d_64 runs the same four
        # permutations numerically at a size that fits.
        "xfail_numerical": {
            "reason": ("ktir_cpu charges the whole [128, 64, 128] fp16 tile "
                       "(2 MiB) to a 2 MiB LX scratchpad; fixed when ktir_cpu "
                       "streams or does not charge a load's result in full"),
            "raises": MemoryError,
            "strict": True,
        },
    },
    "permute3d_64": {
        "base": "permute3d",
        "summary": "The permute3d permutations of a [64, 64, 64] fp16 tensor.",
        "params": {
            ("A", "B", "C", "P0", "P1", "P2", "OA", "OB", "OC"): [
                permute_row((64, 64, 64), perm)
                for perm in ((0, 2, 1), (2, 1, 0), (2, 0, 1), (1, 2, 0))
            ],
            "X_LAYOUT": [stick_layout(3)],
            "Y_LAYOUT": [stick_layout(3)],
        },
        "xfail_numerical": None,
    },
}
