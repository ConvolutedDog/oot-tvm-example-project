from tvm.relax.ty import *


s0 = ShapeType(ndim=2)
print(s0)

objtype = ObjectType()
print(objtype)

dyn_tensor_type = TensorType(ndim=2, dtype="float32")
print(dyn_tensor_type)


f0 = PackedFuncType()
print(f0)