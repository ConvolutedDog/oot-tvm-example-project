from tvm.relax.expr_functor import PyExprMutator, PyExprVisitor
from tvm import relax


# If we want to visit a specific node, we must define a visitor class inheriting from PyExprVisitor and override the related method.
# The methods that can be overridden are listed in `PyExprVisitor._tvm_metadata`.

# The decorator `@relax.expr_functor.visitor` is the key design to make the visitor class a TVM object,
# it uses all the methods and fields of `ASTVisitor` as initial arguments to create a TVM object `_PyExprVisitor`.
# In this example, we can just simply think that `ASTVisitor = _PyExprVisitor(None,visit_constant_)`.

# C++ side: C++ class `PyExprVisitor` is constructed by the function `MakePyExprVisitor` with the arguments of all the methods and fields of python class `ASTVisitor`
# (for example, `visit_constant_` is implicitly converted to `f_visit_constant_`).
# After initialization, `_PyExprVisitor`(also ASTVisitor) is binded with the C++ class `PyExprVisitor`(see also docs/Analysis-of-TVM-python-cpp-binding-mechanism.md).

# You can just use visitor to visit the expr, but you cannot modify the expr.
# If you want to modify the expr, you need to use `PyExprMutator`.
@relax.expr_functor.visitor
class ASTVisitor(PyExprVisitor):
    def __init__(self):
        super().__init__()

    def visit_constant_(self, expr):
        print(f"Visiting expr: {expr}")
        return expr


cons = relax.const(1, dtype="int32")
astVisitor = ASTVisitor()

# depatched to visit_constant_
astVisitor.visit_expr(cons)
print(cons)



# Use ASTMutator to modify the constant node, e.g., change its value to 2
@relax.expr_functor.mutator
class ASTMutator(PyExprMutator):
    def visit_constant_(self, expr):
        print(f"Mutating expr: {expr}")
        # Replace the constant value with 2
        return relax.const(2, dtype=expr.data.dtype)

astMutator = ASTMutator()
new_cons = astMutator.visit_expr(cons)
print("Original constant:", cons)
print("Mutated constant:", new_cons)