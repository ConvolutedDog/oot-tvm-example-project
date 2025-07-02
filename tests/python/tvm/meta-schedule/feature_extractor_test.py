import tvm
import tvm.script.tir as T
import tvm.meta_schedule as ms
from tvm.meta_schedule import TuneContext, MeasureCandidate

@T.prim_func
def matmul(
    A: T.Buffer((512, 512), "float32"), # type: ignore
    B: T.Buffer((512, 512), "float32"), # type: ignore
    C: T.Buffer((512, 512), "float32"), # type: ignore
) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    # body
    # with T.block("root")
    for i0, i1, i2 in T.grid(512, 512, 512):
        with T.block("C"):
            i, j, k = T.axis.remap("SSR", [i0, i1, i2])
            T.reads(C[i, j], A[i, k], B[k, j])
            T.writes(C[i, j])
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
        return sch

    extractor = ms.feature_extractor.PerStoreFeature()
    (feature, ) = extractor.extract_from(
        TuneContext(target="llvm", num_threads=1),
        candidates=[MeasureCandidate(manual_schedule(), [])],
    )

    feature = feature.numpy()
    print(feature)


if __name__ == "__main__":
    test_per_store_feature()