## Inheritance Diagram

### tvm::RelaxExpr

```mermaid
flowchart LR
    tvm::BaseExpr --> tvm::RelaxExpr
    tvm::BaseExpr --> tvm::PrimExpr

    tvm::RelaxExpr --> tvm::GlobalVar
    tvm::RelaxExpr --> tvm::BaseFunc
    tvm::RelaxExpr --> tvm::Op
    tvm::RelaxExpr --> tvm::relax::Call
    tvm::RelaxExpr --> tvm::relax::If
    tvm::RelaxExpr --> tvm::relax::LeafExpr
    tvm::RelaxExpr --> tvm::relax::SeqExpr
    tvm::RelaxExpr --> tvm::relax::Tuple
    tvm::RelaxExpr --> tvm::relax::TupleGetItem

    tvm::BaseFunc --> tvm::relax::Function
    tvm::BaseFunc --> tvm::relax::ExternFunc
    tvm::BaseFunc --> tvm::tir::PrimFunc

    tvm::relax::LeafExpr --> tvm::relax::Constant
    tvm::relax::LeafExpr --> tvm::relax::DataTypeImm
    tvm::relax::LeafExpr --> tvm::relax::PrimValue
    tvm::relax::LeafExpr --> tvm::relax::ShapeExpr
    tvm::relax::LeafExpr --> tvm::relax::StringImm
    tvm::relax::LeafExpr --> tvm::relax::Var

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

### tvm::Type

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

### tvm::transform::Pass

```mermaid
flowchart LR
    tvm::transform::Pass --> tvm::transform::Sequential
```

### tvm::tir::Stmt

```mermaid
flowchart LR
    tvm::tir::Stmt --> tvm::tir::LetStmt
    tvm::tir::Stmt --> tvm::tir::AttrStmt
    tvm::tir::Stmt --> tvm::tir::AssertStmt
    tvm::tir::Stmt --> tvm::tir::BufferStore
    tvm::tir::Stmt --> tvm::tir::BufferRealize
    tvm::tir::Stmt --> tvm::tir::ProducerStore
    tvm::tir::Stmt --> tvm::tir::ProducerRealize
    tvm::tir::Stmt --> tvm::tir::Allocate
    tvm::tir::Stmt --> tvm::tir::AllocateConst
    tvm::tir::Stmt --> tvm::tir::DeclBuffer
    tvm::tir::Stmt --> tvm::tir::Evaluate
    tvm::tir::Stmt --> tvm::tir::SeqStmt
    tvm::tir::Stmt --> tvm::tir::IfThenElse
    tvm::tir::Stmt --> tvm::tir::For
    tvm::tir::Stmt --> tvm::tir::While
    tvm::tir::Stmt --> tvm::tir::Prefetch
    tvm::tir::Stmt --> tvm::tir::Block
    tvm::tir::Stmt --> tvm::tir::BlockRealize
```

### tvm::te::Operation

```mermaid
flowchart LR
    tvm::te::Operation --> tvm::te::PlaceholderOp
    tvm::te::Operation --> tvm::te::ExternOp
    tvm::te::Operation --> tvm::te::ComputeOp
    tvm::te::Operation --> tvm::te::ScanOp
```

### tvm::tir::ExprFunctor\<FType\> & vm::tir::StmtFunctor\<FType\>

```mermaid
flowchart LR
    tvm::tir::ExprFunctor&ltFType&gt --> tvm::tir::ExprVisitor
    tvm::tir::StmtFunctor&ltFType&gt --> tvm::tir::StmtVisitor
    tvm::tir::ExprVisitor --> tvm::tir::StmtExprVisitor
    tvm::tir::StmtVisitor --> tvm::tir::StmtExprVisitor
    tvm::tir::StmtVisitor --> tvm::tir::SRefTreeCreator
    tvm::tir::StmtExprVisitor --> tvm::tir::BufferAxisGraphExtractor

    tvm::tir::ExprFunctor&ltFType&gt --> tvm::tir::ExprMutator
    tvm::tir::StmtFunctor&ltFType&gt --> tvm::tir::StmtMutator
    tvm::tir::ExprMutator --> tvm::tir::StmtExprMutator
    tvm::tir::StmtMutator --> tvm::tir::StmtExprMutator
    tvm::tir::StmtExprMutator --> tvm::tir::DataTypeLegalizer
    tvm::tir::DataTypeLegalizer --> tvm::tir::IndexDataTypeRewriter
    tvm::tir::IndexDataTypeRewriter --> tvm::tir::IndexDataTypeNormalizer
```

### tvm::GlobalInfo

```mermaid
flowchart LR
    tvm::GlobalInfo --> tvm::VDevice
    tvm::GlobalInfo --> tvm::DummyGlobalInfo
    tvm::GlobalInfo --> tvm::relax::distributed::DeviceMesh
```

### tvm::Attrs

```mermaid
flowchart LR
    tvm::Attrs --> tvm::DictAttrs
```

### tvm::PrimExprConvertible

```mermaid
flowchart LR
    tvm::PrimExprConvertible --> tvm::tir::BufferRegion
    tvm::PrimExprConvertible --> tvm::tir::DataProducer
    tvm::PrimExprConvertible --> tvm::tir::IterVar

    tvm::tir::BufferRegion --> tvm::tir::MemCpyDetails

    tvm::tir::DataProducer --> tvm::te::Tensor
```

### tvm::Span

```mermaid
flowchart LR
    tvm::Span --> tvm::SequentialSpan
```

### tvm::ObjectPath

```mermaid
flowchart LR
    tvm::ObjectPath --> tvm::ArrayIndexPath
    tvm::ObjectPath --> tvm::AttributeAccessPath
    tvm::ObjectPath --> tvm::MapValuePath
    tvm::ObjectPath --> tvm::MissingArrayElementPath
    tvm::ObjectPath --> tvm::MissingMapEntryPath
    tvm::ObjectPath --> tvm::RootPath
    tvm::ObjectPath --> tvm::UnknownattributeAccessPath
```

### tvm::BaseValueEqual

```mermaid
flowchart LR
    tvm::BaseValueEqual --> tvm::StructuralEqual
```

### tvm::SEqualReducer::Handler

```mermaid
flowchart LR
    tvm::SEqualReducer::Handler --> tvm::SEqualHandlerDefault
```

### tvm::BaseValueHash

```mermaid
flowchart LR
    tvm::BaseValueHash --> tvm::StructuralHash
```

### tvm::SHashReducer::Handler

```mermaid
flowchart LR
    tvm::SHashReducer::Handler --> tvm::SHashHandlerDefault
```

### tvm::AttrRegistryMap\<KeyType, ValueType\>

```mermaid
flowchart LR
    tvm::AttrRegistryMap&ltKeyType,ValueType&gt --> tvm::TargetKindAttrMap&ltValueType&gt
```

### tvm::tir::BlockScope

```mermaid
flowchart LR
    tvm::tir::BlockScope --> tvm::tir::BlockInfo
```

### Others
- Under namespace `tvm::script`
- Under namespace `tvm::relax`
- Under namespace `tvm::runtime`
- Under namespace `tvm::meta_schedule`
- Under namespace `tvm::ffi`
- Under namespace `tvm::detail`
- Under namespace `tvm::support`
