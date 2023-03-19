//===--------------- Offload.h - CUDA Offloading ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements classes required for offloading to CUDA devices.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_INTERPRETER_OFFLOAD_H
#define LLVM_CLANG_LIB_INTERPRETER_OFFLOAD_H

#include "IncrementalParser.h"

namespace clang {

class DeviceCodeInlinerAction;

class IncrementalCUDADeviceParser : public IncrementalParser {
public:
  IncrementalCUDADeviceParser(std::unique_ptr<CompilerInstance> Instance,
                              llvm::LLVMContext &LLVMCtx, llvm::StringRef Arch,
                              llvm::StringRef FatbinFile, llvm::Error &Err);

  llvm::Expected<PartialTranslationUnit &>
  Parse(llvm::StringRef Input) override;

  // Generate PTX for the last PTU
  llvm::Expected<llvm::StringRef> GeneratePTX();

  // Write last PTX to the fatbinary file
  llvm::Error WriteFatbinary() const;

  ~IncrementalCUDADeviceParser();

protected:
  int SMVersion;
  std::string FatbinFilePath;
  llvm::SmallString<1024> PTXCode;
};

} // namespace clang

#endif // LLVM_CLANG_LIB_INTERPRETER_OFFLOAD_H
