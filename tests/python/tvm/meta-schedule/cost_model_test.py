import tvm
from tvm.script import tir as T
from tvm.tir.schedule import Schedule

from tvm.meta_schedule import TuneContext, MeasureCandidate
from tvm.meta_schedule.runner import RunnerResult
from tvm.meta_schedule.cost_model import PyCostModel, RandomModel
from tvm.meta_schedule.utils import derived_object
from typing import List, Union, Optional

import numpy as np


@tvm.script.ir_module
class MatmulModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:  # pylint: disable=no-self-argument
        T.func_attr({"global_symbol": "matmul", "tir.noalias": True})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        C = T.match_buffer(c, (1024, 1024), "float32")

        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def test_custom_cost_model():

    # custom your own cost model by inheriting from PyCostModel
    @derived_object
    class MyCostModel(PyCostModel):
        def load(self, path: str) -> None:
            pass

        def save(self, path: str) -> None:
            pass

        def update(
            self,
            context: TuneContext,
            candidates: List[MeasureCandidate],
            results: List[RunnerResult],
        ) -> None:
            pass

        def predict(self, context: TuneContext, candidates: List[MeasureCandidate]) -> np.ndarray:
            # always return a random array of length 10 indicating the cost of each candidate
            return np.random.rand(10)
        
    my_cost_model = MyCostModel()

    # NOTE `MeasureCandidate` represents a scheduled IRModule(or operaotr) together with arguemnts infos,
    # that is ready for performance measurement.
    res = my_cost_model.predict(TuneContext(), [MeasureCandidate(Schedule(MatmulModule), []) for i in range(10)])
    print(res)

# TODO low priority
def test_RandomModel():
    model = RandomModel(seed=100)

def test_XGBModel():
    pass


if __name__ == "__main__":
    test_custom_cost_model()
