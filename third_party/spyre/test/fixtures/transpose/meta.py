"""SIGNATURE + VARIANTS + reference oracle + input generator for transpose.

``y = x.permute(perm)`` of a stick-tiled fp16 tensor, one whole tensor per
launch (``grid = [1]``, no distribution loop). Nothing else is computed, so
each kernel is a pure data-movement copy.

What the cases exercise. A permutation that moves the stick dim (the
innermost, contiguous one) is a *re-stick*: the output's stick dim is a
different logical dim from the input's. The ``ktir_cpu`` tier runs the
``ktir`` artifact, which is still logical, so it checks that each kernel
lowers and permutes correctly; a ``compiles_to_binary`` variant is also
compiled and launched on the device by ``test_device_launch.py``.

Shapes. Every dim that is, or becomes, the stick dim is a multiple of the
stick (64 fp16 elements). The rank-2 rows cover square, both non-square
orientations, several blocks along each dim, and an odd block count. The
rank-3 rows are one transpose of the two inner dims, one of the outer two
(the stick dim moves to dim 0), and both 3-cycles; ``permute3d_keep_stick``
swaps the two outer dims and leaves the stick dim alone. ``permute4d`` /
``permute5d`` add ranks 4 and 5 with size-2 batch dims, which need not be
stick multiples because they never become the stick dim. Rank 4 has a
transpose of the inner pair under two batch dims and a 4-cycle; rank 5 has
one permutation that keeps the stick dim innermost and one that moves it to
dim 0.

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

#: Names of the logical input dims of the rank-3, -4 and -5 kernels, in
#: order; a kernel of rank r takes the first r.
_DIM_NAMES = ("d0", "d1", "d2", "d3", "d4")


def permute_constexprs(rank: int) -> list:
    """The constexpr argument names of ``permute{rank}d_stick``, in kernel
    order: the input shape, the permutation, the output shape, the layouts."""
    dims = list(_DIM_NAMES[:rank])
    return (dims + [f"P{i}" for i in range(rank)] + [f"output_{d}" for d in dims]
            + ["X_LAYOUT", "Y_LAYOUT"])


def permute_signature(rank: int) -> dict:
    """``SIGNATURE`` of ``permute{rank}d_stick``: two fp16 pointers, then
    every :func:`permute_constexprs` name."""
    return {"x_ptr": "*fp16", "y_ptr": "*fp16",
            **{name: "constexpr" for name in permute_constexprs(rank)}}


SIGNATURE_3D = permute_signature(3)

#: The module-level signature the loader requires; ``permute3d`` overrides it.
SIGNATURE = SIGNATURE_2D

#: Elements in one stick of the fp16 tensors these kernels move.
_STICK = sticksize(SIGNATURE_2D, "x_ptr")

#: Largest magnitude of an input value; fp16 represents every integer up to
#: 2048 exactly, so the integers in [-_VALUE_RANGE, _VALUE_RANGE] survive the
#: round trip unchanged.
_VALUE_RANGE = 1000


def stick_layout(rank: int) -> tuple:
    """The default stick-tiled layout of a rank-*rank* tensor, labelled, as
    ``tl.spyre_tensor_layout`` takes it.

    The last logical dim is cut into sticks; the stick index sits where the
    rule below puts it and the first logical dim moves just outside the lane:
    rank 2 ``[R, C] -> [C // S, R, C % S]``, rank 3
    ``[d0, d1, d2] -> [d1, d2 // S, d0, d2 % S]``, and in general, for every
    rank ``r >= 2``, ``[d1, ..., d(r-2), d(r-1) // S, d0, d(r-1) % S]``.
    """
    if rank < 2:
        raise ValueError(f"stick_layout needs rank >= 2, got {rank}")
    last = rank - 1
    middle = tuple(range(1, last))
    return ("stick", middle + ((last, "floordiv", _STICK), 0, (last, "mod", _STICK)))


def permute_row(shape: tuple, perm: tuple) -> tuple:
    """One row of a ``permute{r}d`` variant's shape group, for ``r =
    len(shape)``: the input shape, *perm*, then the output shape -- e.g.
    ``(d0, d1, d2, P0, P1, P2, output_d0, output_d1, output_d2)`` at rank 3.
    The output shape is derived so a row cannot disagree with it."""
    if sorted(perm) != list(range(len(shape))):
        raise ValueError(f"{perm} is not a permutation of rank {len(shape)}")
    return (*shape, *perm, *(shape[p] for p in perm))


def permute_group_key(rank: int) -> tuple:
    """The ``params`` key :func:`permute_row` rows of rank *rank* fill: every
    :func:`permute_constexprs` name except the two layouts."""
    return tuple(permute_constexprs(rank)[:-2])


@dataclass(frozen=True)
class Permute(conftest.VariantFactory):
    """Inputs and oracle of a permute of the given rank.

    The shape and permutation come from the variant's swept params:
    ``(M, N)`` and the fixed permutation ``(1, 0)`` at rank 2; at rank
    ``r >= 3``, the first ``r`` of ``d0 .. d4`` and ``P0 .. P(r-1)``.
    """

    rank: int

    def _shape_and_perm(self, combo: dict) -> tuple:
        if self.rank == 2:
            return (combo["M"], combo["N"]), (1, 0)
        shape = tuple(combo[name] for name in _DIM_NAMES[:self.rank])
        perm = tuple(combo[f"P{i}"] for i in range(self.rank))
        return shape, perm

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
            # Square, both non-square orientations, several blocks along each
            # dim in both orientations, and an odd block count (3) along M.
            ("M", "N"): [(64, 64), (128, 64), (64, 128), (128, 256), (256, 128),
                         (192, 64)],
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
        "constexpr": permute_constexprs(3),
        "params": {
            permute_group_key(3): [
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
        # scratchpad, and each of these tensors is 2 MiB of fp16;
        # permute3d_64 runs the same four permutations at a size that fits.
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
            permute_group_key(3): [
                permute_row((64, 64, 64), perm)
                for perm in ((0, 2, 1), (2, 1, 0), (2, 0, 1), (1, 2, 0))
            ],
            "X_LAYOUT": [stick_layout(3)],
            "Y_LAYOUT": [stick_layout(3)],
        },
        "xfail_numerical": None,
    },
    "permute3d_keep_stick": {
        "base": "permute3d",
        "summary": ("y = x.permute(1, 0, 2): the two outer dims swap and the "
                    "stick dim stays innermost."),
        "params": {
            # [64, 128, 64] fp16 is 1 MiB, so it fits ktir_cpu's 2 MiB LX
            # scratchpad (the permute3d xfail does not apply).
            permute_group_key(3): [permute_row((64, 128, 64), (1, 0, 2))],
            "X_LAYOUT": [stick_layout(3)],
            "Y_LAYOUT": [stick_layout(3)],
        },
        "xfail_numerical": None,
    },
    "permute4d": {
        "base": None,
        "summary": "y = x.permute(P0, .., P3) of a rank-4 fp16 stick-tiled tensor.",
        "kernel_fn": kernel.permute4d_stick,
        "SIGNATURE": permute_signature(4),
        "factory": Permute(rank=4),
        "constexpr": permute_constexprs(4),
        "params": {
            permute_group_key(4): [
                # Swap the two inner dims under two size-2 batch dims: the
                # rank-2 transpose with leading dims the copy must carry.
                permute_row((2, 2, 64, 64), (0, 1, 3, 2)),
                # A 4-cycle: input dim 0 becomes the output's stick dim and
                # the input's stick dim lands at dim 2, so every dim moves.
                permute_row((64, 2, 2, 64), (1, 2, 3, 0)),
            ],
            "X_LAYOUT": [stick_layout(4)],
            "Y_LAYOUT": [stick_layout(4)],
        },
        "grid": [1],
        "compiles_to_binary": True,
        "output_key": "y_ptr",
        "rtol": 0.0,
        "atol": 0.0,
        "tags": ["descriptor-load-static", "descriptor-store-static",
                 "simplified:no-loop", "spyre-tensor-layout", "restick"],
    },
    "permute5d": {
        "base": "permute4d",
        "summary": "y = x.permute(P0, .., P4) of a rank-5 fp16 stick-tiled tensor.",
        "kernel_fn": kernel.permute5d_stick,
        "SIGNATURE": permute_signature(5),
        "factory": Permute(rank=5),
        "constexpr": permute_constexprs(5),
        "params": {
            permute_group_key(5): [
                # Reverse the four outer dims; the stick dim stays innermost.
                permute_row((2, 2, 2, 2, 64), (3, 2, 1, 0, 4)),
                # Swap dims 0 and 4 across three size-2 middle dims: the
                # stick dim moves all the way out to dim 0.
                permute_row((64, 2, 2, 2, 64), (4, 1, 2, 3, 0)),
            ],
            "X_LAYOUT": [stick_layout(5)],
            "Y_LAYOUT": [stick_layout(5)],
        },
    },
}
