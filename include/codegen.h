#pragma once

#include "../include/../include/graph.h"
#include <string>
#include <filesystem>


namespace nnc {

// Целевые архитектуры (для llc --march)
enum class TargetArch {
    Native,    // архитектура хост-машины (по умолчанию)
    X86_64,
    AArch64,
    RISCV64,
    WASM32,
};

std::string targetArchToString(TargetArch arch);

// ---- Настройки пайплайна ----
struct CodegenOptions {
    // Пути к инструментам LLVM/MLIR (пустая строка = ищем в PATH)
    std::string mlirOptPath       = "mlir-opt";
    std::string mlirTranslatePath = "mlir-translate";
    std::string llcPath           = "llc";

    // Целевая архитектура для ассемблера
    TargetArch targetArch = TargetArch::Native;

    // Уровень оптимизации LLVM (0-3)
    int optLevel = 2;

    // Рабочая директория (куда сохраняются промежуточные файлы)
    std::filesystem::path workDir = ".";

    // Базовое имя выходных файлов (без расширения)
    std::string outputBaseName = "output";

    // Сохранять ли промежуточные файлы (.mlir, .ll) после компиляции
    bool keepIntermediates = true;

    // Добавлять ли стандартные оптимизации mlir-opt
    // (canonicalize, cse, inline, convert-linalg-to-loops, …)
    bool applyMlirPasses = true;

    // Выводить команды, которые запускаются
    bool verbose = false;

    // Имя функции внутри MLIR/LLVM
    std::string funcName = "forward";
};

// Результат одного шага пайплайна
struct StepResult {
    bool        ok      = false;
    int         retCode = 0;
    std::string command;       // команда, которая была запущена
    std::string outputFile;    // файл, созданный на этом шаге
};

// Итог всего пайплайна
struct CodegenResult {
    StepResult mlirStep;
    StepResult llvmIrStep;
    StepResult asmStep;

    bool allOk() const { return mlirStep.ok && llvmIrStep.ok && asmStep.ok; }

    // Финальный ассемблерный файл
    std::string asmFile() const { return asmStep.outputFile; }
};

// ---- Основной класс пайплайна ----
class Codegen {
public:
    explicit Codegen(CodegenOptions opts = {});

    // Запустить полный пайплайн: Graph -> MLIR -> LLVM IR -> .s
    CodegenResult run(const Graph& graph);

    // Запустить только шаг Graph -> MLIR (удобно для отладки)
    StepResult    emitMLIR(const Graph& graph);

    // Запустить шаги MLIR -> LLVM IR -> .s на существующем .mlir-файле
    CodegenResult compileFromMLIR(const std::filesystem::path& mlirFile);

    const CodegenOptions& options() const { return opts_; }

private:
    CodegenOptions opts_;

    // Шаг 2: mlir-opt + mlir-translate -> .ll
    StepResult runMLIRToLLVM(const std::filesystem::path& mlirFile,
                              const std::filesystem::path& llFile);

    // Шаг 3: llc -> .s
    StepResult runLLCToAsm  (const std::filesystem::path& llFile,
                              const std::filesystem::path& asmFile);

    // Запуск внешней команды с опциональным выводом
    int runCommand(const std::string& cmd) const;

    // Строка флагов оптимизации для mlir-opt
    std::string mlirOptPasses() const;
};

}
