from functools import lru_cache

from pacesense.cbr import CaseBase, MarathonModel
from pacesense.data import synthetic_runners
from pacesense.storage import Repository
from pacesense.study import StudyService


@lru_cache(maxsize=1)
def runners():
    return synthetic_runners(2500, seed=11)


@lru_cache(maxsize=1)
def model():
    return MarathonModel(CaseBase(runners()))


def service(seed: int = 3) -> StudyService:
    return StudyService(model(), runners(), Repository(":memory:"), data_source="test", seed=seed)
