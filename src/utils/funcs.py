import numpy as np
import numpy.typing as npt


def gerono(t: float) -> npt.NDArray[np.floating]:
    if t < 20:
        k = np.pi * t / 10 - np.pi / 2
    else:
        k = 3 * np.pi / 2

    x = -2 * np.sin(k) * np.cos(k)
    y = 2 * (np.sin(k) + 1)

    return np.array([x, y])


def angle_between_vectors(
    vec1: npt.NDArray[np.floating], vec2: npt.NDArray[np.floating]
):
    """
    Will return the counterclockwise angle from vec1 to vec2
    """

    dot = np.dot(vec1, vec2)
    det = np.linalg.det(np.vstack((vec1, vec2)))

    return np.atan2(det, dot)
    # vec1_u = vec1 / np.linalg.norm(vec1)
    # vec2_u = vec2 / np.linalg.norm(vec2)

    # return np.arccos(np.dot(vec1_u, vec2_u))


print(angle_between_vectors(np.array([1, 0]), np.array([1, 0])))
