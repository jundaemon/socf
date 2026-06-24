import time

import numpy as np
from numba import float32, float64, int32, njit, void
from numba.types import Tuple  # type: ignore


@njit(void(int32))
def seed(seed: int) -> None:
    """seeds the numba runtime"""

    np.random.seed(seed)


@njit(float64[:](int32, float32, float32, float32))
def f_1(exp_N: int, T_ns: float, eff: float, lifetime_ns: float) -> np.ndarray:
    """calculates the arrival times generated from pulsing a center, assuming emission to detection is instant"""

    set_t = np.empty(exp_N, dtype=np.float64)
    if eff == 1.0:
        for i in range(exp_N):
            set_t[i] = i * T_ns - lifetime_ns * np.log(np.random.random())
    else:
        sum = 0
        for i in range(exp_N):
            sum += np.floor(np.log(np.random.random()) / np.log(1.0 - eff))
            set_t[i] = sum * T_ns - lifetime_ns * np.log(np.random.random())

    return set_t


@njit(float64[:](float64[:], float64[:]))
def f_2(set_t_1: np.ndarray, set_t_2: np.ndarray) -> np.ndarray:
    return np.sort(np.concat((set_t_1, set_t_2)))


@njit(Tuple((float64[:], float64[:]))(float64[:]))
def f_3(set_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """simulates a beam splitter"""

    mask = np.random.random(len(set_t)) <= 0.5
    return set_t[mask], set_t[~mask]


if __name__ == "__main__":
    seed(10)

    start = time.perf_counter()
    set_t_1 = f_1(500_000, 50.0, 0.5, 3.0)
    set_t_2 = f_1(500_000, 50.0, 0.5, 3.0)
    set_t = f_2(set_t_1, set_t_2)
    set_t_1, set_t_2 = f_3(set_t)
    end = time.perf_counter()

    print(end - start)
    print(len(set_t_1))
    print(len(set_t_2))
