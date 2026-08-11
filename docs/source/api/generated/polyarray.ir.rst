polyarray.ir
============

.. automodule:: polyarray.ir

   
   .. rubric:: Module Attributes

   .. autosummary::
   
      DimSource
      Index
      STMT_FN_OPS
      StmtOp
   
   .. rubric:: Functions

   .. autosummary::
   
      allocate_input
      budget_override
      call_subprogram_inline
      cells_sparsity
      cells_use_only_stmt_atoms
      current_budget_override
      dispatch_einsum
      freeze_array
      freeze_array_bulk
      is_builtin_op
      is_dynamic
      probe_direct_eval
      runtime_einsum
      runtime_einsum_multi
      unpack
      vmap
   
   .. rubric:: Classes

   .. autosummary::
   
      AbsOp
      AddOp
      AssertOp
      AxisLenOp
      BlockDiagOp
      BlockRepeatOp
      BulkOut
      CallOp
      ColStackOp
      CompRankOp
      ComposeViaStdOp
      ConcatOp
      Const
      ConstOp
      DetOp
      DimAtom
      DynBlockRepeatOp
      DynEyeOp
      DynEyeTensorOp
      DynZerosOp
      EinsumOp
      EinsumStmtOp
      EmbedOp
      EyeOp
      FirstColsOp
      GSvdFullOp
      GSvdOp
      HStackOp
      IdentityOp
      InputRef
      IntAtomRef
      InvOp
      InvTransposeOp
      KronFreeOp
      KronOp
      LastColsOp
      MetricOrthonormalOp
      MoveaxisOp
      MulAxisDimOp
      NestedVmapClosure
      OutSpec
      OutputRef
      PinvOp
      ProdDimOp
      ProdShapeOp
      Program
      ProjectOp
      Provenance
      QrOp
      RankOp
      RationalRef
      ReshapeOp
      ScaleAxisDimOp
      ScaleByOp
      ScaleOp
      SignOp
      SinvFullOp
      SolveOp
      SqrtOp
      SqrtSpdOp
      Stmt
      SumDimOp
      SumShapeOp
      SvdOp
      SwitchOp
      SymArray
      SymArrayRef
      SymInput
      SymbolEnv
      SymbolicBudget
      TensordotOp
      TransposeOp
      VmapClosure
      WhileOp
   