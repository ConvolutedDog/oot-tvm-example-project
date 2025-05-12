## Inheritance Diagram

### 1. tvm::tvm::RelaxExpr

```mermaid
flowchart LR
    tvm::BaseExpr --> tvm::RelaxExpr
    tvm::BaseExpr --> tvm::PrimExpr

    tvm::RelaxExpr --> tvm::GlobalVar
    tvm::RelaxExpr --> tvm::BaseFunc
    tvm::RelaxExpr --> tvm::Op
    tvm::RelaxExpr --> tvm::Call
    tvm::RelaxExpr --> tvm::If
    tvm::RelaxExpr --> tvm::LeafExpr
    tvm::RelaxExpr --> tvm::SeqExpr
    tvm::RelaxExpr --> tvm::Tuple
    tvm::RelaxExpr --> tvm::TupleGetItem

    tvm::BaseFunc --> tvm::relax::Function
    tvm::BaseFunc --> tvm::relax::ExternFunc
    tvm::BaseFunc --> tvm::tir::PrimFunc

    tvm::LeafExpr --> tvm::relax::Constant
    tvm::LeafExpr --> tvm::relax::DataTypeImm
    tvm::LeafExpr --> tvm::relax::PrimValue
    tvm::LeafExpr --> tvm::relax::ShapeExpr
    tvm::LeafExpr --> tvm::relax::StringImm
    tvm::LeafExpr --> tvm::relax::Var

    tvm::relax::Var --> tvm::relax::DataflowVar

    tvm::PrimExpr --> tvm::FloatImm
    tvm::PrimExpr --> tvm::IntImm
    tvm::PrimExpr --> tvm::arith::IterMapExpr
    tvm::PrimExpr --> tvm::tir::Add
    tvm::PrimExpr --> tvm::tir::And
    tvm::PrimExpr --> tvm::tir::Broadcast
    tvm::PrimExpr --> tvm::tir::BufferLoad
    tvm::PrimExpr --> tvm::tir::Call
    tvm::PrimExpr --> tvm::tir::Cast
    tvm::PrimExpr --> tvm::tir::Div
    tvm::PrimExpr --> tvm::tir::EQ
    tvm::PrimExpr --> tvm::tir::FloorDiv
    tvm::PrimExpr --> tvm::tir::FloorMod
    tvm::PrimExpr --> tvm::tir::GE
    tvm::PrimExpr --> tvm::tir::GT
    tvm::PrimExpr --> tvm::tir::LE
    tvm::PrimExpr --> tvm::tir::LT
    tvm::PrimExpr --> tvm::tir::Let
    tvm::PrimExpr --> tvm::tir::Max
    tvm::PrimExpr --> tvm::tir::Min
    tvm::PrimExpr --> tvm::tir::Mod
    tvm::PrimExpr --> tvm::tir::Mul
    tvm::PrimExpr --> tvm::tir::NE
    tvm::PrimExpr --> tvm::tir::Not
    tvm::PrimExpr --> tvm::tir::Or
    tvm::PrimExpr --> tvm::tir::ProducerLoad
    tvm::PrimExpr --> tvm::tir::Ramp
    tvm::PrimExpr --> tvm::tir::Reduce
    tvm::PrimExpr --> tvm::tir::Select
    tvm::PrimExpr --> tvm::tir::Shuffle
    tvm::PrimExpr --> tvm::tir::StringImm
    tvm::PrimExpr --> tvm::tir::Sub
    tvm::PrimExpr --> tvm::tir::Var

    tvm::IntImm --> tvm::Bool
    tvm::IntImm --> tvm::Integer

    tvm::arith::IterMapExpr --> tvm::arith::IterSplitExpr
    tvm::arith::IterMapExpr --> tvm::arith::IterSumExpr

    tvm::tir::Var --> tvm::tir::SizeVar
```

### 2. tvm::Type

```mermaid
flowchart LR
    tvm::Type --> tvm::PrimType
    tvm::Type --> tvm::PointerType
    tvm::Type --> tvm::TupleType
    tvm::Type --> tvm::FuncType

    tvm::Type --> tvm::relax::ShapeType
    tvm::Type --> tvm::relax::TensorType
    tvm::Type --> tvm::relax::ObjectType
    tvm::Type --> tvm::relax::PackedFuncType    
```

