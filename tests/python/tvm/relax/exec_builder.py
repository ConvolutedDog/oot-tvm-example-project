from tvm.relax import ExecBuilder

eb = ExecBuilder()
with eb.function("func0", num_inputs=2):
    eb.emit_call("test.vm.add", args=[eb.r(0), eb.r(1)], dst=eb.r(2))
    eb.emit_ret(eb.r(2))

ex = eb.get()
