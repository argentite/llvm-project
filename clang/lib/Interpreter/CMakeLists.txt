set(LLVM_LINK_COMPONENTS
   core
   native
   MC
   Option
   OrcJit
   Support
   Target
   TargetParser
  )

add_clang_library(clangInterpreter
  IncrementalExecutor.cpp
  IncrementalParser.cpp
  Interpreter.cpp
  Offload.cpp

  DEPENDS
  intrinsics_gen
  ClangDriverOptions

  LINK_LIBS
  clangAST
  clangAnalysis
  clangBasic
  clangDriver
  clangEdit
  clangFrontend
  clangLex
  clangParse
  clangSema
  clangSerialization
  clangCodeGen
  clangFrontendTool
  )
