import tvm
from tvm.script import ir as I
from tvm.script import tir as T

testSuites = {}


def register(func):
    testSuites[func.__name__] = func
    return func


@register
def testLoopPartition():

    @I.ir_module
    class Module:
        @T.prim_func
        def main(n: T.int32, m: T.int32):
            for i, j, k in T.grid(4, n, m):
                if T.likely(i * m + j + k < n):
                    T.evaluate(m)
                else:
                    T.evaluate(n)
    Module.show()

    with tvm.transform.PassContext(config={"tir.LoopPartition": {"partition_const_loop": True}}):
        Module = tvm.tir.transform.LoopPartition()(Module)

    print("Partitioned Module:")
    Module.show()


@register
def testLoopVectorize():
    pass


@register
def testLoopUnroll():

    @I.ir_module
    class Module:
        @T.prim_func
        def main(bufPtr: T.handle):
            n = T.int32(is_size_var=True)
            buffer_1 = T.match_buffer(bufPtr, (n,), "int32")
            for i in range(n, n + (n + 2 - n)):
                for j in T.unroll(8):
                    buffer_1[j + 1] = buffer_1[i] + 1

    Module.show()
    with tvm.transform.PassContext(config={"tir.UnrollLoop": {"auto_max_step": 16}}):
        ret = tvm.tir.transform.UnrollLoop()(Module)

    ret.show()


@register
def testInjectVirtualThread():

    @T.prim_func
    def main():
        for i in range(100):
            vthread = T.launch_thread("vthread", 2)
            vthread_1 = T.launch_thread("vthread", 2)
            A = T.allocate([4], "float32", "shared")
            B = T.allocate([4], "float32", "shared")
            C = T.allocate([4], "float32", "shared")

            A_1 = T.Buffer((4,), data=A, scope="shared")
            A_1[vthread] = T.Cast("float32", vthread) + T.float32(1.0)

            B_1 = T.Buffer((4,), data=B, scope="shared")
            B_1[vthread_1] = T.Cast("float32", vthread_1) + T.float32(1.0)

            T.call_extern("int32", "Run", T.tvm_access_ptr(T.type_annotation("float32"), A, 0, 4, 1), T.tvm_access_ptr(
                T.type_annotation("float32"), B, 0, 4, 1), T.tvm_access_ptr(T.type_annotation("float32"), C, 0, 4, 3))

    mod = tvm.IRModule.from_expr(main)
    mod.show()

    newMod = tvm.tir.transform.InjectVirtualThread()(mod)
    newMod.show()


@register
def testRemoveNoOp():

    @T.prim_func
    def main(A: T.Buffer((10, 10), "float32")):
        for i, j in T.grid(10, 10):
            if (i + j < 16):
                T.evaluate(0)  # nop
            else:
                A[i, j] = 1

    mod = tvm.IRModule.from_expr(main)
    mod.show()
    ret = tvm.tir.transform.RemoveNoOp()(mod)["main"].body
    ret.show()


@register
def testSimplify():

    @I.ir_module
    class Module:
        @T.prim_func(private=True)
        def main(A: T.Buffer((16,), "int32")):
            n = tvm.te.var("n", "int32")
            with T.LetStmt(10, var=n):
                c = 6
                for i in T.serial(0, n):
                    A[i] = n + c

    Module.show()

    body = tvm.tir.transform.Simplify()(Module)
    body.show()


if __name__ == "__main__":
    print("🚀🚀🚀Running test suites:")

    for __name, __func in testSuites.items():
        print(f"⏳⏳⏳Running test suite: {__name}")
        __func()
        print(f"✅✅✅Test suite: {__name} passed")
