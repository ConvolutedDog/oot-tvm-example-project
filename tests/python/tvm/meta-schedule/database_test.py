import tvm
import tvm.script.tir as T
from tvm.meta_schedule.database import Database, TuningRecord, Workload
import tvm.meta_schedule as ms
import os
from typing import List, Optional


FUNC = {}


def register(func):
    FUNC[func.__name__] = func
    return func


@tvm.script.ir_module
class MatmulModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:  # pylint: disable=no-self-argument
        T.func_attr({"global_symbol": "matmul", "tir.noalias": True})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        C = T.match_buffer(c, (1024, 1024), "float32")

        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def manual_schedule_matmul(sch: tvm.tir.Schedule):
    block = sch.get_block("matmul")
    i, j, k = sch.get_loops(block=block)
    i_tiles = [1, 1, 2, 512]
    j_tiles = [1, 512, 1, 2]
    k_tiles = [256, 4]
    i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=i_tiles)
    j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=j_tiles)
    k_0, k_1 = sch.split(loop=k, factors=k_tiles)
    sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)
    return sch


@tvm.script.ir_module
class Relu:
    @T.prim_func
    def main(x: T.handle, y: T.handle):
        T.func_attr({"global_symbol": "relu", "tir.noalias": True})
        X = T.match_buffer(x, (1024, 1024), "float32")
        Y = T.match_buffer(y, (1024, 1024), "float32")
        for i, j in T.grid(1024, 1024):
            with T.block("relu"):
                vi, vj = T.axis.remap("SS", [i, j])
                Y[vi, vj] = T.max(X[vi, vj], 0.0)


def manual_schedule_relu(sch: tvm.tir.Schedule):
    block = sch.get_block("relu")
    i, j = sch.get_loops(block=block)

    j_outer, j_inner = sch.split(j, factors=[None, 8])

    sch.parallel(i)
    sch.vectorize(j_inner)
    sch.reorder(i, j_outer, j_inner)

    return sch


@register
def test_Workload():
    workload0 = Workload(MatmulModule)
    workload0.mod.show()


@register
def test_TuningRecord():

    sch = tvm.tir.Schedule(MatmulModule)
    sch = manual_schedule_matmul(sch)
    sch.mod.show()
    sch.trace.show()

    record = TuningRecord(
        trace=sch.trace,
        workload=Workload(MatmulModule)
    )
    print(record.as_json())
    print(record.as_measure_candidate())


@register
def test_json_database():
    # see also: json_database.py

    cur_path = os.path.dirname(os.path.abspath(__file__))
    tmpdir = os.path.join(cur_path, "tune_tmp")

    # Set the path to the database json files.
    path_workload = os.path.join(tmpdir, "workloads.json")
    path_tuning_record = os.path.join(tmpdir, "tuning_records.json")

    db = ms.database.JSONDatabase(
        path_workload, path_tuning_record, module_equality="structural")

    mod = MatmulModule
    # commit the module to the database and return the related workload
    workload = db.commit_workload(mod)
    workload.mod.show()

    sch = tvm.tir.Schedule(mod)
    sch = manual_schedule_matmul(sch)
    record = TuningRecord(
        trace=sch.trace,
        workload=workload,
        run_secs=[1.0],
        target=tvm.target.Target("llvm"),
        args_info=ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
    )

    # commit the record to the database
    db.commit_tuning_record(record)
    # Now you can see the json files in the `tune_tmp` directory.
    # tune_tmp/workloads.json
    # tune_tmp/tuning_records.json

    assert len(db) > 0
    assert db.has_workload(mod)

    # Insert another workload and record
    mod1 = Relu
    workload1 = db.commit_workload(mod1)
    workload1.mod.show()
    sch1 = tvm.tir.Schedule(mod1)
    sch1 = manual_schedule_relu(sch1)
    record1 = TuningRecord(
        trace=sch1.trace,
        workload=workload1,
        run_secs=[1.0],
        target=tvm.target.Target("llvm"),
        args_info=ms.arg_info.ArgInfo.from_prim_func(func=mod1["main"]),
    )
    db.commit_tuning_record(record1)
    assert len(db) > 1
    assert db.has_workload(mod1)

    # Get the top 3 records for the workload
    ret = db.get_top_k(workload, 3)
    print(ret)


@register
def test_memory_database():
    # MemoryDatabase is a in-memory database that stores the tuning records in memory(instead of json file).
    db1 = ms.database.MemoryDatabase()
    mod = MatmulModule
    workload = db1.commit_workload(mod)

    sch = tvm.tir.Schedule(mod)
    sch = manual_schedule_matmul(sch)

    def commit_record(db: ms.database.Database, run_sec: float):
        db.commit_tuning_record(
            TuningRecord(
                trace=sch.trace,
                workload=workload,
                run_secs=[run_sec],
                target=tvm.target.Target("llvm"),
                args_info=ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
            )
        )

    commit_record(db1, 1.0)
    commit_record(db1, 0.5)
    commit_record(db1, 0.3)
    commit_record(db1, 0.2)

    # get the best record for the workload
    best_record = db1.query_tuning_record(
        mod, tvm.target.Target("llvm"), "main")
    print(best_record.run_secs)


@register
def test_PyDatabase():
    from tvm.meta_schedule.database import PyDatabase
    # We can inherit from PyDatabase to customize the database.

    # Must use @ms.utils.derived_object to decorate(register) the class.
    @ms.utils.derived_object
    class MyMemoryDatabase(PyDatabase):
        def __init__(self):
            super().__init__()
            self.tuning_records_ = []
            self.workloads_ = []

        def has_workload(self, mod: tvm.ir.IRModule) -> bool:
            for workload in self.workloads_:
                if tvm.ir.structural_equal(mod, workload.mod):
                    return True
            return False

        def commit_workload(self, mod: tvm.ir.IRModule) -> Workload:
            if self.has_workload(mod):
                for workload in self.workloads_:
                    if tvm.ir.structural_equal(mod, workload.mod):
                        return workload
            else:
                workload = Workload(mod)
                self.workloads_.append(workload)
                return workload
            
        def commit_tuning_record(self, record: TuningRecord) -> None:
            self.tuning_records_.append(record)

        def get_all_tuning_records(self) -> List[TuningRecord]:
            return self.tuning_records_

        def get_top_k(self, workload: Workload, top_k: int) -> List[TuningRecord]:
            return sorted(
                list(
                    filter(
                        lambda x: tvm.ir.structural_equal(workload.mod, x.workload.mod),
                        self.tuning_records_,
                    )
                ),
                key=lambda x: sum(x.run_secs) / len(x.run_secs) if x.run_secs else 1e9,
            )[:top_k]
        
        def __len__(self) -> int:
            return len(self.tuning_records_)
        
        def query_tuning_record(
            self, mod: tvm.ir.IRModule, target: tvm.target.Target, workload_name: Optional[str] = None
        ) -> Optional[TuningRecord]:
            if self.has_workload(mod):
                records = self.get_top_k(self.commit_workload(mod), 2)
                if len(records) == 1:
                    return records[0]
                elif len(records) == 2:
                    return records[1]
            return None
        
        def query_schedule(
            self, mod: tvm.ir.IRModule, target: tvm.target.Target, workload_name: Optional[str] = None
        ) -> Optional[tvm.tir.Schedule]:
            record = self.query_tuning_record(mod, target, workload_name)
            if record is not None:
                sch = tvm.tir.Schedule(record.workload.mod)
                record.trace.apply_to_schedule(sch, remove_postproc=False)
                return sch
            return None

        def query_ir_module(
            self, mod: tvm.ir.IRModule, target: tvm.target.Target, workload_name: Optional[str] = None
        ) -> Optional[tvm.ir.IRModule]:
            record = self.query_tuning_record(mod, target, workload_name)
            if record is not None:
                sch = tvm.tir.Schedule(record.workload.mod)
                record.trace.apply_to_schedule(sch, remove_postproc=False)
                return sch.mod
            return None




if __name__ == "__main__":
    for func in FUNC.values():
        print(f'⭕⭕⭕ {func.__name__} is running...')
        func()
        print(f'✅✅✅ {func.__name__} is done!')
