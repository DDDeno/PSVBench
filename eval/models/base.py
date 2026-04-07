from __future__ import annotations

import abc
import dataclasses
from typing import Any, Dict, Optional, Sequence

from eval.qa_schema import QAItem


@dataclasses.dataclass(frozen=True)
class Prediction:
    answer: Optional[str]
    raw: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


class ModelAdapter(abc.ABC):
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def predict(
        self,
        item: QAItem,
        *,
        frames_data_urls: Optional[Sequence[str]],
        prompt: str,
    ) -> Prediction:
        raise NotImplementedError
