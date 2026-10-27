#!/usr/bin/env python3
"""
Contract test for train_lora.evaluate().

Exists because a change that added per-sample predictions to the metrics dict
broke training after ~1 h of compute: the per-epoch logger rounds every value in
that dict, and round() raises on a numpy array. The --smoke-test path returns
before the epoch loop, so it did not catch it.

Asserts the two properties the logger depends on:
  1. evaluate(...)                      -> dict of SCALARS only (round-able)
  2. evaluate(..., return_preds=True)   -> (metrics, preds, labels)

Run (lora env):  python training/test_evaluate_contract.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.train_lora import evaluate  # noqa: E402


class _Out:
    def __init__(self, logits, loss):
        self.logits, self.loss = logits, loss


class _StubModel:
    """Minimal stand-in: returns fixed logits and a scalar loss."""
    def eval(self):
        return self

    def __call__(self, **kw):
        n = kw["labels"].shape[0]
        return _Out(torch.randn(n, 2), torch.tensor(0.5))


def _loader(n_batches=3, bs=4):
    for _ in range(n_batches):
        yield {"labels": torch.randint(0, 2, (bs,)),
               "input_ids": torch.ones(bs, 8, dtype=torch.long)}


def main() -> int:
    fails = []

    m = evaluate(_StubModel(), list(_loader()), "cpu", True)
    if not isinstance(m, dict):
        fails.append(f"default call returned {type(m)}, expected dict")
    else:
        for k, v in m.items():
            if not isinstance(v, (int, float)):
                fails.append(f"metrics['{k}'] is {type(v).__name__}, not a scalar")
            else:
                try:
                    round(v, 6)                      # the exact call that broke
                except TypeError as e:
                    fails.append(f"round(metrics['{k}']) raised: {e}")

    got = evaluate(_StubModel(), list(_loader()), "cpu", True, return_preds=True)
    if not (isinstance(got, tuple) and len(got) == 3):
        fails.append("return_preds=True did not return a 3-tuple")
    else:
        _, preds, labels = got
        if not isinstance(preds, np.ndarray) or not isinstance(labels, np.ndarray):
            fails.append("return_preds=True did not return numpy arrays")
        elif len(preds) != len(labels):
            fails.append(f"preds/labels length mismatch: {len(preds)} vs {len(labels)}")

    if fails:
        print("FAILED:")
        for f in fails:
            print("  " + f)
        return 1
    print(f"OK - metrics keys {sorted(m)} are all scalars; "
          f"return_preds=True yields (metrics, preds, labels)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
