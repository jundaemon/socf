import numpy as np
from numba import float64, int64, njit, prange, void
from numba.types import Tuple  # type: ignore


@njit(void(int64))
def seed_env(seed: int) -> None:
    np.random.seed(seed)


@njit(float64[:](int64, float64, float64, float64))
def f_1(exp_N: int, T_ns: float, eff: float, lifetime_ns: float) -> np.ndarray:
    # the vectorized version of this is comparable in speed but uses twice the memory
    set_t = np.empty(exp_N, dtype=np.float64)
    if eff == 1.0:
        for i in range(exp_N):
            set_t[i] = i * T_ns - lifetime_ns * np.log(np.random.random())
    else:
        sum = -1
        for i in range(exp_N):
            # geometric distribution to find the number of pulses for next emission
            sum += np.floor(np.log(np.random.random()) / np.log(1.0 - eff)) + 1
            set_t[i] = sum * T_ns - lifetime_ns * np.log(np.random.random())

    return set_t


@njit(float64[:](float64[:], float64[:]))
def f_2(set_t_1: np.ndarray, set_t_2: np.ndarray) -> np.ndarray:
    set_t = np.empty(len(set_t_1) + len(set_t_2), dtype=np.float64)
    i = 0
    j = 0

    # didn't use compiled np.sort because it was taking 10s to sort a size 1_000_000 array
    for k in range(len(set_t)):
        if i == len(set_t_1):
            set_t[k:] = set_t_2[j:]
            break

        if j == len(set_t_2):
            set_t[k:] = set_t_1[i:]
            break

        if set_t_1[i] <= set_t_2[j]:
            set_t[k] = set_t_1[i]
            i += 1
        else:
            set_t[k] = set_t_2[j]
            j += 1

    return set_t


@njit(Tuple((float64[:], float64[:]))(float64[:]))
def f_3(set_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.random.random(len(set_t)) <= 0.5
    return set_t[mask], set_t[~mask]


@njit(float64[:](float64[:], float64[:], float64))
def f_4(set_t_1: np.ndarray, set_t_2: np.ndarray, half_window_ns: float) -> np.ndarray:
    starts = np.empty(len(set_t_1), dtype=np.int64)
    ends = np.empty(len(set_t_1), dtype=np.int64)

    # faster than np.searchsorted because this exploits sets being sorted thus not resetting indices
    size = 0
    ptr_1 = 0
    ptr_2 = 0
    for i in range(len(set_t_1)):
        while ptr_1 < len(set_t_2) and set_t_2[ptr_1] < set_t_1[i] - half_window_ns:
            ptr_1 += 1

        if ptr_2 < ptr_1:
            ptr_2 = ptr_1

        while ptr_2 < len(set_t_2) and set_t_2[ptr_2] < set_t_1[i] + half_window_ns:
            ptr_2 += 1

        starts[i] = ptr_1
        ends[i] = ptr_2
        size += ptr_2 - ptr_1

    taus = np.empty(size, dtype=np.float64)
    i = 0
    for j in range(len(set_t_1)):
        for k in range(starts[j], ends[j]):
            taus[i] = set_t_1[j] - set_t_2[k]
            i += 1

    return taus


@njit(Tuple((int64[:], float64))(float64[:], int64, float64))
def f_5(taus: np.ndarray, bins: int, T_ns: float) -> tuple[np.ndarray, float]:
    hist, edges = np.histogram(taus, bins=bins)
    return hist, np.floor(T_ns / (edges[1] - edges[0]))


@njit(Tuple((int64[:], int64))(int64[:], float64))
def f_6(hist: np.ndarray, bpp: float) -> tuple[np.ndarray, int]:
    # potential peak if bin is larger than neighbours and above threshold height
    side_peaks_i = (
        np.where(
            (hist[1:-1] > hist[2:])
            & (hist[1:-1] > hist[:-2])
            & (hist[1:-1] > hist.max() * 0.8)
        )[0]
        + 1
    )
    # peaks should be roughly size of pulses apart
    side_peaks_i = side_peaks_i[
        np.concat((np.full(1, True), np.diff(side_peaks_i) > np.floor(bpp * 0.9)))
    ]

    return side_peaks_i[1:-1], len(hist) // 2


@njit(float64(int64[:], float64, int64[:], int64))
def f_7(
    hist: np.ndarray, bpp: float, side_peaks_i: np.ndarray, tau_zero_i: int
) -> float:
    # g^2(0) = area of peak at tau = 0 / average area of side peaks
    areas = np.empty(len(side_peaks_i))
    for i in range(len(side_peaks_i)):
        areas[i] = hist[side_peaks_i[i] - bpp // 2 : side_peaks_i[i] + bpp // 2].sum()

    return hist[tau_zero_i - bpp // 2 : tau_zero_i + bpp // 2].sum() / areas.mean()


@njit(float64[:](float64[:], float64[:], int64), parallel=True)
def label_gen(eff_1s: np.ndarray, eff_2s: np.ndarray, seed: int) -> np.ndarray:
    g2_zeros = np.empty(len(eff_1s), dtype=np.float64)

    exp_N = 500_000
    T_ns = 50.0
    lifetime_ns = 3.0
    half_window_ns = 250.0
    bins = 10_000
    seed_env(seed)

    for i in prange(len(eff_1s)):  # type: ignore
        set_t_1 = f_1(exp_N, T_ns, eff_1s[i], lifetime_ns)
        set_t_2 = f_1(exp_N, T_ns, eff_2s[i], lifetime_ns)
        set_t = f_2(set_t_1, set_t_2)
        set_t_1, set_t_2 = f_3(set_t)
        taus = f_4(set_t_1, set_t_2, half_window_ns)
        hist, bpp = f_5(taus, bins, T_ns)
        side_peaks_i, tau_zero_i = f_6(hist, bpp)
        g2_zeros[i] = f_7(hist, bpp, side_peaks_i, tau_zero_i)

    return g2_zeros


@njit(int64[:, :](float64[:], float64[:], int64), parallel=True)
def sample_gen(eff_1s: np.ndarray, eff_2s: np.ndarray, seed: int) -> np.ndarray:
    histograms = np.empty((len(eff_1s), 500), dtype=np.int64)

    exp_N = 50
    T_ns = 50.0
    lifetime_ns = 3.0
    half_window_ns = 250.0
    bins = 500
    seed_env(seed)

    for i in prange(len(eff_1s)):  # type: ignore
        set_t_1 = f_1(exp_N, T_ns, eff_1s[i], lifetime_ns)
        set_t_2 = f_1(exp_N, T_ns, eff_2s[i], lifetime_ns)
        set_t = f_2(set_t_1, set_t_2)
        set_t_1, set_t_2 = f_3(set_t)
        taus = f_4(set_t_1, set_t_2, half_window_ns)
        hist, _ = f_5(taus, bins, T_ns)
        histograms[i,] = hist

    return histograms


if __name__ == "__main__":
    pass
