import tvm
from tvm import relax as rx
from tvm import tir

test = {}


def register(func):
    test[func.__name__] = func
    return func


@register
def test_block_builder():
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")

    x = rx.Var("x", rx.TensorStructInfo([m, n], "float32"))
    y = rx.Var("y", rx.TensorStructInfo([m, n], "float32"))

    bb = rx.BlockBuilder()

    with bb.function("func", [x, y]):
        with bb.dataflow():
            lv0 = bb.emit(rx.op.add(x, y))
            assert lv0.name_hint == "lv"
            gv0 = bb.emit_output(lv0)

        bb.emit_func_output(gv0)

    mod = bb.finalize()
    func = mod["func"]

    mod.show()
    func.show()


if __name__ == "__main__":
    for name, func in test.items():
        print(f"Running test: {name}")
        func()
        print(f"Test {name} passed.\n")

    print("All tests passed.")
    print("Test block_builder completed successfully.")
