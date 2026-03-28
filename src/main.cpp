#include "../include/../include/graph.h"
#include "../include/../include/codegen.h"
#include <iostream>
#include <string>
#include <vector>
#include <cstring>

// ============================================================
// main.cpp — Драйвер nnc-codegen
//
// Использование:
//   nnc-codegen [опции] <input.mlir>
//
// Опции:
//   --mlir-opt    <path>     Путь к mlir-opt
//   --mlir-trans  <path>     Путь к mlir-translate
//   --llc         <path>     Путь к llc
//   --arch        <arch>     Целевая архитектура: native|x86_64|aarch64|riscv64|wasm32
//   -O<n>                    Уровень оптимизации (0..3, по умолчанию 2)
//   --out         <name>     Базовое имя выходных файлов (без расширения)
//   --outdir      <dir>      Рабочая директория
//   --no-passes              Не применять passes mlir-opt
//   --no-intermediate        Удалять промежуточные файлы (.ll, lowered.mlir)
//   --func        <name>     Имя генерируемой функции (по умолчанию "forward")
//   -v, --verbose            Выводить запускаемые команды
//   -h, --help               Показать эту справку
//
// При отсутствии входного .mlir-файла запускается демо-режим:
// создаётся тестовый граф (Add + Relu + MatMul) и компилируется.
// ============================================================

static void printHelp(const char* argv0) {
    std::cout
        << "nnc-codegen: Neural Network Compiler — MLIR/LLVM backend\n"
        << "\nUsage:\n"
        << "  " << argv0 << " [options] [input.mlir]\n"
        << "\nOptions:\n"
        << "  --mlir-opt   <path>   Path to mlir-opt    (default: mlir-opt)\n"
        << "  --mlir-trans <path>   Path to mlir-translate (default: mlir-translate)\n"
        << "  --llc        <path>   Path to llc          (default: llc)\n"
        << "  --arch       <arch>   Target: native|x86_64|aarch64|riscv64|wasm32\n"
        << "  -O<n>                 Optimization level 0..3 (default: 2)\n"
        << "  --out        <name>   Output base name     (default: output)\n"
        << "  --outdir     <dir>    Output directory     (default: .)\n"
        << "  --no-passes           Skip mlir-opt lowering passes\n"
        << "  --no-intermediate     Delete .ll and lowered.mlir after compilation\n"
        << "  --func       <name>   MLIR function name   (default: forward)\n"
        << "  -v, --verbose         Print executed commands\n"
        << "  -h, --help            Show this help\n"
        << "\nExamples:\n"
        << "  # Run demo (synthetic graph)\n"
        << "  " << argv0 << " --verbose\n"
        << "\n"
        << "  # Compile existing MLIR to native assembly\n"
        << "  " << argv0 << " model.mlir --out model --verbose\n"
        << "\n"
        << "  # Cross-compile to AArch64\n"
        << "  " << argv0 << " model.mlir --arch aarch64 --out model_arm\n"
        << "\n"
        << "  # Use custom LLVM install\n"
        << "  " << argv0 << " model.mlir \\\n"
        << "      --mlir-opt /opt/llvm/bin/mlir-opt \\\n"
        << "      --mlir-trans /opt/llvm/bin/mlir-translate \\\n"
        << "      --llc /opt/llvm/bin/llc\n";
}

// Разбор флага -O<n>
static int parseOptLevel(const std::string& s) {
    if (s == "-O0") return 0;
    if (s == "-O1") return 1;
    if (s == "-O2") return 2;
    if (s == "-O3") return 3;
    return -1;
}

static nnc::TargetArch parseArch(const std::string& s) {
    if (s == "x86_64"  || s == "x86-64") return nnc::TargetArch::X86_64;
    if (s == "aarch64" || s == "arm64")   return nnc::TargetArch::AArch64;
    if (s == "riscv64")                   return nnc::TargetArch::RISCV64;
    if (s == "wasm32")                    return nnc::TargetArch::WASM32;
    return nnc::TargetArch::Native;
}

// ---- Синтетический демо-граф для тестирования без ONNX ----
// Граф: input → Add(input, input) → Relu → MatMul(relu, weight) → output
static nnc::Graph makeDemoGraph() {
    nnc::Graph g;
    g.name = "demo_graph";

    // Входной тензор [1, 128]
    nnc::TensorInfo input;
    input.name  = "input";
    input.dtype = nnc::DType::Float32;
    input.shape.dims = {1, 128};
    g.addTensor(input);
    g.inputs.push_back("input");

    // Инициализатор (вес) [128, 64]
    nnc::TensorInfo weight;
    weight.name  = "weight";
    weight.dtype = nnc::DType::Float32;
    weight.shape.dims = {128, 64};
    weight.isInitializer = true;
    g.addTensor(weight);

    // Промежуточные тензоры
    auto makeTensor = [&](const std::string& name,
                          std::vector<int64_t> dims) {
        nnc::TensorInfo t;
        t.name  = name;
        t.dtype = nnc::DType::Float32;
        t.shape.dims = std::move(dims);
        g.addTensor(t);
    };
    makeTensor("add_out",    {1, 128});
    makeTensor("relu_out",   {1, 128});
    makeTensor("output",     {1, 64});
    g.outputs.push_back("output");

    // Узел Add
    nnc::Node addNode;
    addNode.name   = "add_0";
    addNode.opType = nnc::OpType::Add;
    addNode.inputs  = {"input", "input"};
    addNode.outputs = {"add_out"};
    g.nodes.push_back(addNode);

    // Узел Relu
    nnc::Node reluNode;
    reluNode.name   = "relu_0";
    reluNode.opType = nnc::OpType::Relu;
    reluNode.inputs  = {"add_out"};
    reluNode.outputs = {"relu_out"};
    g.nodes.push_back(reluNode);

    // Узел MatMul
    nnc::Node mmNode;
    mmNode.name   = "matmul_0";
    mmNode.opType = nnc::OpType::MatMul;
    mmNode.inputs  = {"relu_out", "weight"};
    mmNode.outputs = {"output"};
    g.nodes.push_back(mmNode);

    return g;
}

// ---- main ----
int main(int argc, char* argv[]) {
    nnc::CodegenOptions opts;
    std::string inputMlir;

    // Разбор аргументов командной строки
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            printHelp(argv[0]);
            return 0;
        }
        else if (arg == "-v" || arg == "--verbose") {
            opts.verbose = true;
        }
        else if (arg == "--no-passes") {
            opts.applyMlirPasses = false;
        }
        else if (arg == "--no-intermediate") {
            opts.keepIntermediates = false;
        }
        else if (arg == "--mlir-opt" && i+1 < argc) {
            opts.mlirOptPath = argv[++i];
        }
        else if (arg == "--mlir-trans" && i+1 < argc) {
            opts.mlirTranslatePath = argv[++i];
        }
        else if (arg == "--llc" && i+1 < argc) {
            opts.llcPath = argv[++i];
        }
        else if (arg == "--arch" && i+1 < argc) {
            opts.targetArch = parseArch(argv[++i]);
        }
        else if (arg == "--out" && i+1 < argc) {
            opts.outputBaseName = argv[++i];
        }
        else if (arg == "--outdir" && i+1 < argc) {
            opts.workDir = argv[++i];
        }
        else if (arg == "--func" && i+1 < argc) {
            opts.funcName = argv[++i];
        }
        else if (int lvl = parseOptLevel(arg); lvl >= 0) {
            opts.optLevel = lvl;
        }
        else if (arg[0] != '-') {
            inputMlir = arg;
        }
        else {
            std::cerr << "Unknown option: " << arg
                      << "  (use --help for usage)\n";
            return 1;
        }
    }

    nnc::Codegen codegen(opts);
    nnc::CodegenResult result;

    if (inputMlir.empty()) {
        // Демо-режим: строим граф в памяти и компилируем
        std::cout << "No input .mlir specified — running demo graph.\n";
        nnc::Graph demoGraph = makeDemoGraph();
        demoGraph.dump(/*verbose=*/true);
        std::cout << "\n";
        result = codegen.run(demoGraph);
    } else {
        // Компиляция существующего .mlir-файла
        result = codegen.compileFromMLIR(inputMlir);
    }

    // ---- Отчёт о результатах ----
    std::cout << "\n=== Compilation results ===\n";
    auto printStep = [](const std::string& name, const nnc::StepResult& s) {
        std::cout << "  " << name << ": "
                  << (s.ok ? "OK" : "FAILED")
                  << "  →  " << s.outputFile << "\n";
    };
    printStep("MLIR emission", result.mlirStep);
    printStep("LLVM IR       ", result.llvmIrStep);
    printStep("Assembly      ", result.asmStep);

    if (result.allOk()) {
        std::cout << "\nSuccess! Assembly: " << result.asmFile() << "\n";
        return 0;
    } else {
        std::cerr << "\nCompilation failed.\n";
        return 1;
    }
}
