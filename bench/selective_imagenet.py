"""ImageNet selective classification: `RiskControl` against Angelopoulos (2026), Section 3.1.

Conformal Risk Control for Non-Monotonic Losses, arXiv:2602.20151. A ResNet-152 predicts
when its top softmax probability is above a threshold, and the loss
`1{error, predicted} - alpha 1{predicted} + alpha` has mean at most alpha exactly when
the selective error is at most alpha. The loss is not monotone in the threshold.

The paper's data is `imagenet/imagenet-resnet152.npz` in the directory of its Google
Drive archive (see github.com/aangelopoulos/nonmonotonic-crc):

uv run python bench/selective_imagenet.py ~/.cache/calibre-bench/nonmonotonic-crc/data

Each replicate draws a calibration set and checks three rules on it:
- `calibre`: `RiskControl` on a grid of the calibration confidences.
- `reference`: the same corrected rule written directly from the loss table. It must
  give the same threshold as `calibre` in every replicate.
- `paper`: the CRC of the paper's code (`SelectiveClassifier.fit`), no correction.
The paper reports CRC selective accuracy "a hair below 90%" and a stability estimate
of 0.006. Its text says n = 1000, but its notebook uses n = 200 and prints 0.0052; the
same estimator gives about 0.001 at n = 1000.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from calibre import Feedback, Loss, RiskControl, Signed

ALPHA = 0.1
CALIBRATION_SIZES = (200, 1000)
REPLICATES = 100
BOOTSTRAPS = 50


class SelectiveLoss(Loss):
    """`1{error, predicted} - alpha 1{predicted} + alpha`, as a loss of `Signed` bounds.

    The point is minus the confidence and the target is the error indicator, so the
    upper bound `threshold - confidence` is negative exactly when the classifier
    predicts.
    """

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    @property
    def maximum(self) -> float:
        return 1.0

    def loss(
        self, lower: np.ndarray, upper: np.ndarray, target: np.ndarray, censored: np.ndarray
    ) -> np.ndarray:
        predicted = upper < 0
        return target * predicted - self.alpha * predicted + self.alpha


def load(directory: str) -> tuple[np.ndarray, np.ndarray]:
    """Top softmax probability `[S]` and error indicator `[S]` of each validation image."""
    data = np.load(Path(directory) / "imagenet" / "imagenet-resnet152.npz")
    softmax, labels = data["smx"], data["labels"].astype(int)
    return softmax.max(axis=1).astype(np.float64), (softmax.argmax(axis=1) != labels)


def calibre_threshold(confidence: np.ndarray, error: np.ndarray, grid: np.ndarray) -> float:
    """Threshold of `RiskControl` with one node, one column, and one row per image."""
    n = len(confidence)
    feedback = Feedback(
        origin=np.arange(n),
        column=np.zeros(n, dtype=np.int64),
        point=-confidence[:, None],
        target=error[:, None].astype(np.float64),
        issued=np.zeros((n, 1)),
        censored=np.zeros((n, 1), dtype=bool),
        score=Signed(),
    )
    calibrator = RiskControl(SelectiveLoss(ALPHA), grid, ALPHA)
    state = calibrator.update(calibrator.initial_state(1, 1), feedback)
    return float(calibrator.threshold(state)[0, 0])


def reference_threshold(confidence: np.ndarray, error: np.ndarray, grid: np.ndarray) -> float:
    """Smallest grid threshold with `(loss sum + 1) / (n + 1) <= alpha`, else inf."""
    predicted = confidence[:, None] > grid[None, :]  # [n, G]
    losses = error[:, None] * predicted - ALPHA * predicted + ALPHA
    feasible = losses.sum(axis=0) + 1 <= ALPHA * (len(confidence) + 1)
    return float(grid[np.argmax(feasible)]) if feasible.any() else np.inf


def paper_threshold(confidence: np.ndarray, error: np.ndarray) -> float:
    """`SelectiveClassifier.fit` of the paper's code: uncorrected CRC at data points."""
    order = np.argsort(confidence)[::-1]
    weight = np.where(error[order], -(1 - ALPHA), ALPHA)
    feasible = np.nonzero(1 + np.cumsum(weight) >= 0)[0]
    return float(confidence[order[feasible[-1]]]) if len(feasible) else 1.0


def paper_stability(confidence: np.ndarray, error: np.ndarray, rng: np.random.Generator) -> float:
    """The paper's `estimate_beta_def` for its CRC: bootstrap leave-one-out loss change."""
    n = len(confidence)
    deltas = []
    for _ in range(BOOTSTRAPS):
        index = rng.choice(n, size=n + 1, replace=True)
        p, e = confidence[index], error[index]
        star = paper_threshold(p, e)
        keep = ~np.eye(n + 1, dtype=bool)
        loo = np.array([paper_threshold(p[keep[i]], e[keep[i]]) for i in range(n + 1)])
        loss_loo = e * (p > loo) - ALPHA * (p > loo) + ALPHA
        loss_star = e * (p > star) - ALPHA * (p > star) + ALPHA
        deltas.append((loss_loo - loss_star).mean())
    return max(0.0, float(np.mean(deltas)))


def evaluate(threshold: float, confidence: np.ndarray, error: np.ndarray) -> dict:
    """Selective accuracy, prediction rate, and mean loss on the held-out images."""
    predicted = confidence > threshold
    accuracy = 1 - error[predicted].mean() if predicted.any() else np.nan
    loss = (error * predicted - ALPHA * predicted + ALPHA).mean()
    return {"threshold": threshold, "accuracy": accuracy, "rate": predicted.mean(), "risk": loss}


def replicate(confidence: np.ndarray, error: np.ndarray, n: int, rng: np.random.Generator):
    """Rows of one random calibration split, one per rule."""
    calibration = np.zeros(len(confidence), dtype=bool)
    calibration[rng.choice(len(confidence), size=n, replace=False)] = True
    p, e = confidence[calibration], error[calibration]
    held_p, held_e = confidence[~calibration], error[~calibration]
    grid = np.unique(p)
    thresholds = {
        "calibre": calibre_threshold(p, e, grid),
        "reference": reference_threshold(p, e, grid),
        "paper": paper_threshold(p, e),
    }
    return [{"rule": rule, **evaluate(t, held_p, held_e)} for rule, t in thresholds.items()]


def main(directory: str) -> None:
    confidence, error = load(directory)
    rng = np.random.default_rng(0)
    print(f"{len(confidence)} images, top-1 accuracy {1 - error.mean():.4f}, alpha {ALPHA}")
    for n in CALIBRATION_SIZES:
        rows = []
        for index in range(REPLICATES):
            rows += [{"replicate": index, **row} for row in replicate(confidence, error, n, rng)]
        runs = pd.DataFrame(rows)
        by_rule = runs.pivot(index="replicate", columns="rule", values="threshold")
        mismatches = int((by_rule["calibre"] != by_rule["reference"]).sum())
        table = runs.groupby("rule")[["threshold", "accuracy", "rate", "risk"]].agg(["mean", "std"])
        table[("accuracy", "share >= 0.9")] = (
            runs.assign(ok=runs["accuracy"] >= 1 - ALPHA).groupby("rule")["ok"].mean()
        )
        sample = rng.choice(len(confidence), size=n, replace=False)
        beta = paper_stability(confidence[sample], error[sample], rng)
        print(f"\nn = {n}, {REPLICATES} replicates")
        print(f"calibre and reference thresholds differ in {mismatches} replicates")
        print(f"paper CRC stability estimate beta = {beta:.4f}")
        print(table.round(4).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
