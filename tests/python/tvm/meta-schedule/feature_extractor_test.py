import tvm
import tvm.script
import tvm.script.tir as T
import tvm.meta_schedule as ms
from tvm.meta_schedule import TuneContext, MeasureCandidate


@T.prim_func
def matmul(
    A: T.Buffer((512, 512), "float32"),  # type: ignore
    B: T.Buffer((512, 512), "float32"),  # type: ignore
    C: T.Buffer((512, 512), "float32"),  # type: ignore
) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    # body
    # with T.block("root")
    for i0, i1, i2 in T.grid(512, 512, 512):
        with T.block("C"):
            i, j, k = T.axis.remap("SSR", [i0, i1, i2])
            with T.init():
                C[i, j] = T.float32(0)
            C[i, j] = C[i, j] + A[i, k] * B[k, j]


def test_per_store_feature():
    def manual_schedule():
        func = matmul
        sch = tvm.tir.Schedule(func, debug_mask="all")
        block = sch.get_block("C")
        i, j, k = sch.get_loops(block)
        i_o, i_i = sch.split(i, factors=[None, 16])  # outer: 32
        j_o, j_i = sch.split(j, factors=[None, 8])  # outer: 64
        sch.reorder(i_o, j_o, k, j_i, i_i)
        sch.vectorize(j_i)
        sch.parallel(i_o)
        sch.parallel(j_o)
        sch.unroll(k)
        # sch.mod.show()
        return sch

    extractor = ms.feature_extractor.PerStoreFeature()
    feature = extractor.extract_from(
        TuneContext(target="llvm", num_threads=1),
        candidates=[MeasureCandidate(manual_schedule(), [])],
    )

    print(feature[0])
    # `feature` include 164 features of the `BufferStoreNode`(`C[i, j] = C[i, j] + A[i, k] * B[k, j]` in this case)
    # If there is another `BufferStoreNode` in the IRModule,
    # e.g. `D[i, j] = C[i, j] + T.float32(3.14)`, `feature` will include 164 features of the `BufferStoreNode`(`D[i, j]`)
    # feature.shape will be (2,164)
    print(feature[0].shape)

    # Next, we will break down the meaning of some features

    # Scheduled IRModule
    # for i0_0 in T.parallel(32):
    #     for i1_0 in T.parallel(64):
    #         for i2 in T.unroll(512):
    #             for i1_1 in T.vectorized(8):
    #                 for i0_1 in range(16):
    #                     with T.block("C"):
    #                         i = T.axis.spatial(512, i0_0 * 16 + i0_1)
    #                         j = T.axis.spatial(512, i1_0 * 8 + i1_1)
    #                         k = T.axis.reduce(512, i2)
    #                         T.reads(A[i, k], B[k, j])
    #                         T.writes(C[i, j])
    #                         with T.init():
    #                             C[i, j] = T.float32(0.0)
    #                         C[i, j] = C[i, j] + A[i, k] * B[k, j]

    f = feature[0].numpy()
    # See also `tests/python/meta_schedule/test_meta_schedule_feature_extractor_per_store_feature.py`


def test_per_store_feature_with_multiple_BufferStoreNode():
    @tvm.script.ir_module
    class myIRModule:
        @T.prim_func
        def mulAdd(
            A: T.Buffer((512, 512), "float32"),  # type: ignore
            B: T.Buffer((512, 512), "float32"),  # type: ignore
            C: T.Buffer((512, 512), "float32"),  # type: ignore
            D: T.Buffer((512, 512), "float32"),  # type: ignore
        ) -> None:
            T.func_attr({"global_symbol": "mulAdd", "tir.noalias": True})
            for i0, i1, i2 in T.grid(512, 512, 512):
                with T.block("C"):
                    i, j, k = T.axis.remap("SSR", [i0, i1, i2])
                    with T.init():
                        C[i, j] = T.float32(0)
                    C[i, j] = C[i, j] + A[i, k] * B[k, j]
                    D[i, j] = C[i, j] + T.float32(3.14)

        @T.prim_func
        def relu(
            x: T.Buffer((512, 512), "float32"),  # type: ignore
            y: T.Buffer((512, 512), "float32"),  # type: ignore
        ) -> None:
            T.func_attr({"global_symbol": "relu", "tir.noalias": True})
            for i0, i1 in T.grid(512, 512):
                with T.block("relu"):
                    i, j = T.axis.remap("SS", [i0, i1])
                    with T.init():
                        y[i, j] = T.float32(0)
                    y[i, j] = T.max(x[i, j], T.float32(0))

    sch = tvm.tir.Schedule(myIRModule, debug_mask="all")
    extractor = ms.feature_extractor.PerStoreFeature()
    feature = extractor.extract_from(
        TuneContext(target="llvm", num_threads=1),
        candidates=[MeasureCandidate(sch, [])],
    )
    print(feature)  # [runtime.NDArray(0x2xxxx)]
    print(feature[0].numpy().shape)  # (3, 164)


if __name__ == "__main__":
    test_per_store_feature()
    test_per_store_feature_with_multiple_BufferStoreNode()
