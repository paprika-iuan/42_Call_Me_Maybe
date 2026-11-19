import numpy as np
from typing import Any, cast


class LogitsManager:
    def __init__(self, logits: list[float]) -> None:
        self.rawLogits = np.array(logits, dtype=float)
        self._processedLogits = np.copy(self.rawLogits)
        self._rng = np.random.default_rng()

    def _get_probabilities(self, logits: Any) -> np.ndarray:
        exp_logits = np.exp(logits - np.max(logits))
        return cast(np.ndarray, exp_logits / np.sum(exp_logits))

    def _apply_temperature(self, tempVar: float) -> None:
        if tempVar > 0:
            self._processedLogits /= tempVar

    def _apply_topk(self, k: int) -> None:
        if k >= len(self._processedLogits):
            return

        descarted_indexes = np.argsort(self._processedLogits)[:-k]
        self._processedLogits[descarted_indexes] = float('-inf')

    def _apply_topp(self, p: float) -> None:
        if p >= 1.0:
            return
        sorted_indexes = np.argsort(self._processedLogits)[::-1]
        probs = self._get_probabilities(self._processedLogits[sorted_indexes])

        cumsum = np.cumsum(probs)
        indexes_to_remove = cumsum > p
        indexes_to_remove[1:] = indexes_to_remove[:-1].copy()
        indexes_to_remove[0] = False
        indexes_to_discard = sorted_indexes[indexes_to_remove]
        self._processedLogits[indexes_to_discard] = float('-inf')

    def _apply_softmax(self) -> None:
        self._processedLogits = self._get_probabilities(self._processedLogits)

    def _apply_multinominal(self) -> int:
        self._processedLogits = np.cumsum(self._processedLogits)
        closer_index = np.int64(
            np.searchsorted(self._processedLogits, self._rng.random())
            )
        return cast(int, closer_index)

    def filter_logits(self, temperature: float, topk: int, topp: float) -> int:
        self._apply_temperature(temperature)
        self._apply_topk(topk)
        self._apply_topp(topp)
        self._apply_softmax()
        probable_index = self._apply_multinominal()
        return probable_index

    @property
    def processed_logits(self) -> np.ndarray:
        if self._processedLogits is None:
            raise ("No processed logits available.")
        return self._processedLogits
