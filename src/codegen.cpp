#include "../include/../include/codegen.h"
#include "../include/../include/mlir_emitter.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <cstdlib>


namespace nnc {

std::string targetArchToString(TargetArch arch) {
    switch (arch) {
        case TargetArch::X86_64:  return "x86-64";
        case TargetArch::AArch64: return "aarch64";
        case TargetArch::RISCV64: return "riscv64";
        case TargetArch::WASM32:  return "wasm32";
        case TargetArch::Native:
        default:                   return "";  // пустая строка = native
    }
}

// ---- Конструктор ----
Codegen::Codegen(CodegenOptions opts) : opts_(std::move(opts)) {
    // Создать рабочую директорию, если не существует
    std::filesystem::create_directories(opts_.workDir);
}

// ---- Запуск команды ----
int Codegen::runCommand(const std::string& cmd) const {
    if (opts_.verbose)
        std::cerr << "[nnc-codegen] $ " << cmd << "\n";
    return std::system(cmd.c_str());
}

// ---- Шаг 1: Graph -> .mlir ----
StepResult Codegen::emitMLIR(const Graph& graph) {
    StepResult res;
    auto mlirFile = opts_.workDir / (opts_.outputBaseName + ".mlir");

    MLIREmitOptions emitOpts;
    emitOpts.funcName           = opts_.funcName;
    emitOpts.emitNodeComments   = true;
    emitOpts.emitModuleWrapper  = true;

    MLIREmitter emitter(emitOpts);

    std::ofstream ofs(mlirFile);
    if (!ofs) {
        std::cerr << "Error: cannot open " << mlirFile << " for writing\n";
        res.ok = false;
        return res;
    }

    try {
        emitter.emit(graph, ofs);
    } catch (const std::exception& e) {
        std::cerr << "Error during MLIR emission: " << e.what() << "\n";
        res.ok = false;
        return res;
    }

    ofs.close();
    res.ok         = true;
    res.outputFile = mlirFile.string();
    if (opts_.verbose)
        std::cerr << "[nnc-codegen] MLIR written to " << mlirFile << "\n";
    return res;
}

// ---- Цепочка пассов mlir-opt ----
std::string Codegen::mlirOptPasses() const {
    if (!opts_.applyMlirPasses) return "";

    return
        " --linalg-bufferize"                  // тензоры -> буферы
        " --convert-linalg-to-loops"           // linalg -> SCF
        " --convert-scf-to-cf"                 // SCF -> CF (Control Flow)
        " --convert-cf-to-llvm"                // CF-> LLVM диалект
        " --convert-arith-to-llvm"             // arith -> LLVM
        " --convert-func-to-llvm"              // func -> llvm.func
        " --reconcile-unrealized-casts"        // удаляем лишние касты
        " --canonicalize"                      // упрощение (CSE, DCE, …)
        ;
}

// ---- Шаг 2: .mlir -> .ll ----
StepResult Codegen::runMLIRToLLVM(const std::filesystem::path& mlirFile,
                                   const std::filesystem::path& llFile) {
    StepResult res;

    // Сначала mlir-opt применяет пассы lowering
    auto optedMlir = opts_.workDir / (opts_.outputBaseName + "_lowered.mlir");

    std::string optCmd = opts_.mlirOptPath
                       + mlirOptPasses()
                       + " " + mlirFile.string()
                       + " -o " + optedMlir.string();
    res.command = optCmd;

    int rc1 = runCommand(optCmd);
    if (rc1 != 0) {
        std::cerr << "mlir-opt failed with code " << rc1 << "\n";
        res.retCode = rc1;
        res.ok      = false;
        return res;
    }

    // mlir-translate конвертирует опущенный MLIR -> LLVM IR
    std::string transCmd = opts_.mlirTranslatePath
                         + " --mlir-to-llvmir"
                         + " " + optedMlir.string()
                         + " -o " + llFile.string();
    res.command += "\n" + transCmd;

    int rc2 = runCommand(transCmd);
    res.retCode  = rc2;
    res.ok       = (rc2 == 0);
    res.outputFile = llFile.string();

    if (!res.ok)
        std::cerr << "mlir-translate failed with code " << rc2 << "\n";
    else if (opts_.verbose)
        std::cerr << "[nnc-codegen] LLVM IR written to " << llFile << "\n";

    // Убираем промежуточный .mlir если не нужен
    if (!opts_.keepIntermediates)
        std::filesystem::remove(optedMlir);

    return res;
}

// ---- Шаг 3: .ll -> .s ----
StepResult Codegen::runLLCToAsm(const std::filesystem::path& llFile,
                                  const std::filesystem::path& asmFile) {
    StepResult res;

    std::string marchFlag;
    std::string arch = targetArchToString(opts_.targetArch);
    if (!arch.empty())
        marchFlag = " --march=" + arch;

    // llc: генерация ассемблера из LLVM IR
    // -filetype=asm - текстовый ассемблер (.s)
    std::string cmd = opts_.llcPath
                    + marchFlag
                    + " -O" + std::to_string(opts_.optLevel)
                    + " -filetype=asm"
                    + " " + llFile.string()
                    + " -o " + asmFile.string();
    res.command = cmd;

    int rc = runCommand(cmd);
    res.retCode  = rc;
    res.ok       = (rc == 0);
    res.outputFile = asmFile.string();

    if (!res.ok)
        std::cerr << "llc failed with code " << rc << "\n";
    else if (opts_.verbose)
        std::cerr << "[nnc-codegen] Assembly written to " << asmFile << "\n";

    return res;
}

// ---- Полный пайплайн ----
CodegenResult Codegen::run(const Graph& graph) {
    CodegenResult result;

    result.mlirStep = emitMLIR(graph);
    if (!result.mlirStep.ok) return result;

    return compileFromMLIR(result.mlirStep.outputFile);
}

CodegenResult Codegen::compileFromMLIR(const std::filesystem::path& mlirFile) {
    CodegenResult result;
    result.mlirStep.ok         = true;
    result.mlirStep.outputFile = mlirFile.string();

    auto llFile  = opts_.workDir / (opts_.outputBaseName + ".ll");
    auto asmFile = opts_.workDir / (opts_.outputBaseName + ".s");

    result.llvmIrStep = runMLIRToLLVM(mlirFile, llFile);
    if (!result.llvmIrStep.ok) return result;

    result.asmStep = runLLCToAsm(llFile, asmFile);

    if (!opts_.keepIntermediates && result.asmStep.ok)
        std::filesystem::remove(llFile);

    return result;
}

} 