/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <cassert>
#include <iostream>
#include <tvm/../../src/relay/ir/indexed_graph.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/function.h>
#include <tvm/relay/parser.h>
#include <unordered_set>

namespace tvm {
namespace relay {
namespace {

// A module stolen from onnx/test_forward.py::test_loop which combines
// functions, recursion, control flow, tuples as well as the usual operator
// calls. We include the known post-dfs indexes in comments to help write the
// tests.
IRModule TestRecursiveIRModule() {
  Device myDevice = {kDLCPU, 0};
  Constant const0(
      runtime::NDArray::Empty(ShapeTuple({1}), DataType::Int(64), myDevice));
  Constant const1(runtime::NDArray::Empty(ShapeTuple({0, 1}),
                                          DataType::Float(32), myDevice));
  Map<String, Array<ObjectRef>> metadata;
  metadata.Set("relay.Constant", Array<ObjectRef>({const0, const1}));
  constexpr const char *kModel = R"(
    #[version = "0.0.5"]
    def @main(%trip_count: int64,                                        // 0
              %cond: bool,                                               // 1
              %y: Tensor[(1), float32])                                  // 2
              -> (Tensor[(1), float32], Tensor[(?, ?), float32]) {
      %17 = (
        let %while_loop = fn (%iter_count: int64,                        // 3
                              %max_count: int64,                         // 4
                              %cond_in: bool,                            // 5
                              %y_in: Tensor[(1), float32],               // 6
                              %scan_out: Tensor[(?, ?), float32])        // 7
                              -> (int64, int64, bool, Tensor[(1), float32], Tensor[(?, ?), float32]) {
          %0 = equal(%cond_in, True);                                    // 11
          %1 = less(%iter_count, %max_count);                            // 13
          %2 = logical_and(%0, %1);                                      // 14
          if (%2) {
            %3 = cast(%iter_count, dtype="float32");                     // 20
            %4 = add(%y_in, %3);                                         // 21
            %5 = less(%4, 5f);                                           // 23
            %6 = squeeze(%5);                                            // 24
            %7 = reshape(%iter_count, newshape=[1]);                     // 29
            %8 = (%7, meta[relay.Constant][0]);                          // 31
            %9 = concatenate(%8);                                        // 32
            %10 = copy(%4);                                              // 36
            %11 = dyn.broadcast_to(%scan_out, %9, shape=None);           // 33
            %12 = expand_dims(%10, axis=0);                              // 37
            %13 = (%11, %12);                                            // 38
            %14 = add(%iter_count, 1i64);                                // 17
            %15 = cast(%6, dtype="bool");                                // 25
            %16 = concatenate(%13);                                      // 39
            %while_loop(%14, %max_count, %15, %4, %16)                   // 40
          } else {
            (%iter_count, %max_count, %cond_in, %y_in, %scan_out)        // 41
          }                                                              // 42
        };                                                               // 43
        %while_loop                                                      // 44
      );                                                                 // 45
      %18 = %17(0i64, %trip_count, %cond, %y, meta[relay.Constant][1]);  // 48
      %19 = %18.3;                                                       // 49
      %20 = %18.4;                                                       // 50
      (%19, %20)                                                         // 51
    }                                                                    // 52
  )";
  return ParseModule("string", kModel, /*init_module=*/{}, metadata);
}

void RecursiveExprRegression() {
  IRModule irMod = TestRecursiveIRModule();
  auto main = Downcast<Function>(irMod->Lookup("main"));
  auto graph = CreateIndexedGraph(main);
  graph->CheckValid();

  {
    // Dataflow node properties for %4
    auto node = graph->index_to_node(21);
    const auto *callNode = node->ref().as<CallNode>();
    assert(callNode != nullptr);
    const auto *opNode = callNode->op.as<OpNode>();
    assert(opNode != nullptr);
    std::cout << "  opNode->name: " << opNode->name << '\n';
    assert(opNode->name == "add");

    // 3 inputs (the op itself is an input)
    assert(node->inputs_.size() == 3);
    assert(node->inputs_[0]->index_ == 15);  // the add op
    assert(node->inputs_[1]->index_ == 6);   // %y_in
    assert(node->inputs_[2]->index_ == 20);  // %3

    // 3 outputs
    assert(node->outputs_.size() == 3);
    assert(node->outputs_[0]->index_ == 23);  // %5
    assert(node->outputs_[1]->index_ == 36);  // %10
    assert(node->outputs_[2]->index_ == 40);  // recursive %while_loop call

    // In the 'if' basic block
    assert(node->basic_block_->index_ == 42);

    // Dominator 'parent' is recursive call
    assert(node->dominator_parent_->index_ == 40);

    // One dominator child from %3
    assert(node->dominator_children_.size() == 1);
    assert(node->dominator_children_[0]->index_ == 20);
  }

  {
    // The recursive call to %while_loop does not depend on %while_loop
    auto node = graph->index_to_node(40);
    const auto *callNode = node->ref().as<CallNode>();
    assert(callNode != nullptr);
    const auto *varNode = callNode->op.as<VarNode>();
    assert(varNode != nullptr);
    std::cout << "  varNode->name_hint(): " << varNode->name_hint() << '\n';
    assert(varNode->name_hint() == "while_loop");

    assert(node->inputs_.size() == 5);
    assert(node->inputs_[0]->index_ == 17);  // %14
    assert(node->inputs_[1]->index_ == 4);   // %max_count
    assert(node->inputs_[2]->index_ == 25);  // %15
    assert(node->inputs_[3]->index_ == 21);  // %4
    assert(node->inputs_[4]->index_ == 39);  // %16
  }

  {
    // Downstream nodes of %18
    auto node = graph->index_to_node(48);
    std::unordered_set<const IndexedGraph<Expr>::Node *> downstreams;
    node->AccumulateDownstreamNodes(&downstreams);
    assert(downstreams.size() == 4);
    for (const auto *downstream : downstreams) {
      assert(downstream->index_ >= 49 && downstream->index_ <= 52);
    }
  }

  {
    // Dominates relation for %4
    auto upstream = graph->index_to_node(21);
    // Path 1: 21->23->24->25->40
    // Path 2: 21->36->37->38->39->40
    // Then 40->43
    auto downstream = graph->index_to_node(43);
    assert(downstream->Dominates(upstream));
  }
}

// A module with unused let-bound function. The 'add' operator should have no
// dominator since it is used both in the unused function and in the main body.
IRModule TestUnusedLetBoundIRModule() {
  constexpr const char *kModel = R"(
    #[version = "0.0.5"]
    def @main(%x: int64) -> int64 {   // 0
      let %f = fn (                   // 5
        %y: int64                     // 1
      ) {
        add(%x, %y)                   // 3
      };
      if (less(%x, 5i64)) {
        add(%x, 3i64)                 // 10
      } else {
        %x
      }
    }
  )";
  return ParseModule("string", kModel);
}

void UnusedLetVars() {
  IRModule irMod = TestUnusedLetBoundIRModule();
  auto main = Downcast<Function>(irMod->Lookup("main"));
  auto graph = CreateIndexedGraph(main);
  graph->CheckValid();

  {
    auto node = graph->index_to_node(2);
    const auto *opNode = node->ref().as<OpNode>();
    assert(opNode);
    std::cout << "  opNode->name: " << opNode->name << '\n';
    assert(opNode->name == "add");
    assert(node->outputs_.size() == 2);
    assert(node->outputs_[0]->index_ == 3);
    assert(node->outputs_[1]->index_ == 10);
    assert(node->dominator_parent_ == nullptr);
  }
}

}  // namespace
}  // namespace relay
}  // namespace tvm
