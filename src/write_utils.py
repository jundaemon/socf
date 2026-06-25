import numpy as np
import polars as pl

from src.calc_src import label_gen

SEEDS = [10, 42, 67]


def write_local_excel(path: str) -> None:
    eff_1s = np.repeat(np.linspace(0.1, 1.0, 91), 91)
    eff_2s = eff_1s.reshape(91, 91).T.ravel()

    labels = []
    for seed in SEEDS:
        print(f"starting calculations for seed: {seed}")
        labels.append(label_gen(eff_1s, eff_2s, seed))

    pl.DataFrame(
        {
            "seeds": np.repeat(SEEDS, len(eff_1s)),
            "eff_1s": np.concat([eff_1s] * 3),
            "eff_2s": np.concat([eff_2s] * 3),
            "labels": np.concat(labels),
        }
    ).write_excel(path, autofit=True)
    print(f"data written locally to {path}")
