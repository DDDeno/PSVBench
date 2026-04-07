from __future__ import annotations

import random
from typing import Optional, Sequence

from eval.models.base import ModelAdapter, Prediction
from eval.qa_schema import QAItem


class RandomChoiceAdapter(ModelAdapter):
    def __init__(self, *, seed: int = 0, name: str = "random") -> None:
        self._rng = random.Random(int(seed))
        self._name = str(name)

    def name(self) -> str:
        return self._name

    def predict(self, item: QAItem, *, frames_data_urls: Optional[Sequence[str]], prompt: str) -> Prediction:
        _ = (item, frames_data_urls, prompt)
        ans = self._rng.choice(["A", "B", "C", "D"])
        return Prediction(answer=ans, raw=ans, meta={"strategy": "random"})

