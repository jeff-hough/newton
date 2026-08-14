# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Index-handling kernels shared by every controller.

Gather and scatter are the two places a controller can silently address the
wrong joint, so both live here rather than being reimplemented per control law.
"""

import warp as wp


@wp.kernel
def _gather_flat_kernel(
    src: wp.array[wp.float32],  # caller-owned array, any length
    indices: wp.array[wp.int32],  # (n,) destination slot -> source slot
    dst: wp.array[wp.float32],  # (n,)
):
    slot = wp.tid()
    dst[slot] = src[indices[slot]]


@wp.kernel
def _scatter_flat_kernel(
    src: wp.array[wp.float32],  # (n,)
    indices: wp.array[wp.int32],  # (n,) source slot -> destination slot
    dst: wp.array[wp.float32],  # caller-owned array, any length
):
    slot = wp.tid()
    dst[indices[slot]] = src[slot]
