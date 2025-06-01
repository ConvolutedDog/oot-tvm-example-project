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

if __name__ == "__main__":
    print("🚀🚀🚀Running test suites:")
    
    for __name, __func in testSuites.items():
        print(f"⏳⏳⏳Running test suite: {__name}")
        __func()
        print(f"✅✅✅Test suite: {__name} passed")
