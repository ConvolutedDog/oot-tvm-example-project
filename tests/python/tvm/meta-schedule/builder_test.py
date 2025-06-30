import tvm
import tvm.script.tir as T
from tvm.meta_schedule.builder import BuilderInput, BuilderResult, Builder, PyBuilder
import os

@tvm.script.ir_module
class MatmulModule:
    @T.prim_func
    def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:  # pylint: disable=no-self-argument
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


def test_builder():
    mod = MatmulModule
    builder_inputs = [BuilderInput(mod, tvm.target.Target("llvm"))]

    builder0 = Builder.create("local")
    builder_results = builder0.build(builder_inputs)
    print(builder_results[0].artifact_path)
    print(builder_results[0].error_msg)

    os.remove(builder_results[0].artifact_path)
    os.rmdir(os.path.dirname(builder_results[0].artifact_path))


if __name__ == "__main__":
    test_builder()