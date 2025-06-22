# NOTE `pipeline` is a sequence of `transform.Pass`.
# The file `pipeline.py` pre-defines some pipelines.
# `zero_pipeline` is the basic pipeline that applies some fundamental passes.
# `default_build_pipeline` is the default pipeline of `tvm.compile`
# `static_shape_tuning_pipeline` is for tuning models with static shapes.

import tvm

from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R

import numpy as np


def test_zero_pipeline():

    @tvm.script.ir_module
    class MyModule:
        @R.function
        def matmul(x: R.Tensor((128,128), "float32"), y: R.Tensor((128,128), "float32")):
            z = R.matmul(x, y)
            return z

        @R.function
        def relu(z: R.Tensor((128,128), "float32")):
            z_relu = R.nn.relu(z)
            return z_relu


    from tvm.relax.pipeline import zero_pipeline
    mod = zero_pipeline()(MyModule)
    # `zero_pipeline` will lower(legalize) the relax.function to tir.prim_func
    print(mod)

# TODO @Benkangpeng: Finish the rest test after reading the source code of transform.py
def test_default_build_pipeline():
    pass

def test_static_shape_tuning_pipeline():
    pass


if __name__ == "__main__":
    test_zero_pipeline()



