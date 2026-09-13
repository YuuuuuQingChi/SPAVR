"""NumPy rotation helpers using scalar-last (x, y, z, w) quaternions."""

import numpy as np


def normalize_quaternion(quaternion):
    quaternion = np.asarray(quaternion, dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)


def quaternion_conjugate(quaternion):
    quaternion = np.asarray(quaternion, dtype=np.float64)
    result = quaternion.copy()
    result[..., :3] *= -1.0
    return result


def quaternion_multiply(left, right):
    """Hamilton product for arrays whose final dimension is ``xyzw``."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_xyz, left_w = left[..., :3], left[..., 3:4]
    right_xyz, right_w = right[..., :3], right[..., 3:4]
    xyz = (
        left_w * right_xyz
        + right_w * left_xyz
        + np.cross(left_xyz, right_xyz)
    )
    w = left_w * right_w - np.sum(left_xyz * right_xyz, axis=-1, keepdims=True)
    return np.concatenate([xyz, w], axis=-1)


def rotation_matrix_to_quaternion(matrix):
    """Convert rotation matrices to scalar-last (x, y, z, w) quaternions.

    The branch on the largest diagonal element keeps the conversion stable for
    camera rotations close to 180 degrees, where the trace-only formula is
    poorly conditioned.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(
            f"rotation matrices must have shape (...,3,3), got {matrix.shape}"
        )

    flat = matrix.reshape(-1, 3, 3)
    quaternion = np.empty((len(flat), 4), dtype=np.float64)
    for index, rotation in enumerate(flat):
        trace = np.trace(rotation)
        if trace > 0.0:
            scale = 2.0 * np.sqrt(max(trace + 1.0, 0.0))
            quaternion[index] = (
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            )
        elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
            scale = 2.0 * np.sqrt(
                max(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2], 0.0)
            )
            quaternion[index] = (
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
            )
        elif rotation[1, 1] > rotation[2, 2]:
            scale = 2.0 * np.sqrt(
                max(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2], 0.0)
            )
            quaternion[index] = (
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
            )
        else:
            scale = 2.0 * np.sqrt(
                max(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1], 0.0)
            )
            quaternion[index] = (
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            )

    return normalize_quaternion(quaternion).reshape(matrix.shape[:-2] + (4,))


def quaternion_to_rotvec(quaternion):
    """Convert scalar-last quaternions to the shortest rotation vectors."""
    quaternion = normalize_quaternion(quaternion)
    quaternion = np.where(quaternion[..., 3:4] < 0.0, -quaternion, quaternion)

    xyz = quaternion[..., :3]
    xyz_norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(xyz_norm, quaternion[..., 3:4])
    scale = np.divide(
        angle,
        xyz_norm,
        out=np.full_like(angle, 2.0),
        where=xyz_norm > 1.0e-12,
    )
    return xyz * scale


def relative_rotvec(current_quaternion, next_quaternion):
    """Return ``Log(R_next R_current^-1)`` as a rotation vector."""
    delta = quaternion_multiply(
        normalize_quaternion(next_quaternion),
        quaternion_conjugate(normalize_quaternion(current_quaternion)),
    )
    return quaternion_to_rotvec(delta)
