import tvm.relax.frontend.nn as nn


def test_visitor():
    class MyMutator(nn.Mutator):
        def visit_module(self, name: str, node: nn.Module) -> any:
            print(f"visit_module: {name}")
            return self.visit(name, node)

        def visit_param(self, name: str, node: nn.Parameter) -> any:
            print(f"visit_param: {name}")
            return self.visit(name, node)

        def visit_effect(self, name: str, node: nn.Effect) -> any:
            print(f"visit_effect: {name}")
            return self.visit(name, node)

        def visit_modulelist(self, name: str, node: nn.ModuleList) -> any:
            print(f"visit_modulelist: {name}")
            return self.visit(name, node)


    linear0 = nn.modules.Linear(10, 10, False)
    relu0 = nn.modules.ReLU()

    modulelist0 = nn.ModuleList([linear0, relu0])

    mutator = MyMutator()
    mutator.visit("modulelist0", modulelist0)


if __name__ == "__main__":
    test_visitor()