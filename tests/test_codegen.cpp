#include "../include/test_framework.h"
#include "../include/codegen.h"
#include "../include/graph.h"
#include "../include/mlir_emitter.h"
#include <filesystem>
#include <fstream>
#include <sstream>

// ============================================================
// test_codegen.cpp — Тесты пайплайна компиляции (Codegen)
//
// Покрываем:
//   • TargetArch → строка (все значения)
//   • CodegenOptions: значения по умолчанию
//   • emitMLIR: создаёт файл с корректным содержимым
//   • compileFromMLIR: обработка несуществующего файла
//   • Шаг млир → ллвм: формирование командной строки
//   • Шаг llc: флаги оптимизации, --march
//   • allOk() при частичных неудачах
//   • Рабочая директория создаётся автоматически
// ============================================================

namespace fs = std::filesystem;

// Вспомогательный граф для тестов
static nnc::Graph makeSimpleGraph(const std::string& name = "test_g") {
    nnc::Graph g;
    g.name = name;

    auto mkT = [](const std::string& n, std::vector<int64_t> d) {
        nnc::TensorInfo t;
        t.name = n; t.dtype = nnc::DType::Float32; t.shape.dims = d;
        return t;
    };
    g.addTensor(mkT("x",   {1, 8}));
    g.addTensor(mkT("out", {1, 8}));
    g.inputs  = {"x"};
    g.outputs = {"out"};

    nnc::Node r; r.name = "r"; r.opType = nnc::OpType::Relu;
    r.inputs = {"x"}; r.outputs = {"out"};
    g.nodes.push_back(r);
    return g;
}

// ----------------------------------------------------------------
// Suite: TargetArch
// ----------------------------------------------------------------

TEST(TargetArch, NativeIsEmptyString) {
    CHECK_EQ(nnc::targetArchToString(nnc::TargetArch::Native), std::string(""));
}

TEST(TargetArch, X86_64) {
    CHECK_EQ(nnc::targetArchToString(nnc::TargetArch::X86_64), std::string("x86-64"));
}

TEST(TargetArch, AArch64) {
    CHECK_EQ(nnc::targetArchToString(nnc::TargetArch::AArch64), std::string("aarch64"));
}

TEST(TargetArch, RISCV64) {
    CHECK_EQ(nnc::targetArchToString(nnc::TargetArch::RISCV64), std::string("riscv64"));
}

TEST(TargetArch, WASM32) {
    CHECK_EQ(nnc::targetArchToString(nnc::TargetArch::WASM32), std::string("wasm32"));
}

// ----------------------------------------------------------------
// Suite: CodegenOptions — значения по умолчанию
// ----------------------------------------------------------------

TEST(CodegenOptions, DefaultOptLevel) {
    nnc::CodegenOptions opts;
    CHECK_EQ(opts.optLevel, 2);
}

TEST(CodegenOptions, DefaultToolNames) {
    nnc::CodegenOptions opts;
    CHECK_EQ(opts.mlirOptPath,       std::string("mlir-opt"));
    CHECK_EQ(opts.mlirTranslatePath, std::string("mlir-translate"));
    CHECK_EQ(opts.llcPath,           std::string("llc"));
}

TEST(CodegenOptions, DefaultFuncName) {
    nnc::CodegenOptions opts;
    CHECK_EQ(opts.funcName, std::string("forward"));
}

TEST(CodegenOptions, DefaultTargetIsNative) {
    nnc::CodegenOptions opts;
    CHECK_EQ(opts.targetArch, nnc::TargetArch::Native);
}

TEST(CodegenOptions, DefaultKeepIntermediates) {
    nnc::CodegenOptions opts;
    CHECK(opts.keepIntermediates);
}

TEST(CodegenOptions, DefaultApplyPasses) {
    nnc::CodegenOptions opts;
    CHECK(opts.applyMlirPasses);
}

TEST(CodegenOptions, DefaultVerboseFalse) {
    nnc::CodegenOptions opts;
    CHECK(!opts.verbose);
}

// ----------------------------------------------------------------
// Suite: StepResult
// ----------------------------------------------------------------

TEST(StepResult, DefaultNotOk) {
    nnc::StepResult r;
    CHECK(!r.ok);
    CHECK_EQ(r.retCode, 0);
    CHECK(r.command.empty());
    CHECK(r.outputFile.empty());
}

TEST(CodegenResult, AllOkRequiresAllThree) {
    nnc::CodegenResult r;
    r.mlirStep.ok   = true;
    r.llvmIrStep.ok = true;
    r.asmStep.ok    = true;
    CHECK(r.allOk());
}

TEST(CodegenResult, NotOkIfMlirFailed) {
    nnc::CodegenResult r;
    r.mlirStep.ok   = false;
    r.llvmIrStep.ok = true;
    r.asmStep.ok    = true;
    CHECK(!r.allOk());
}

TEST(CodegenResult, NotOkIfLlvmFailed) {
    nnc::CodegenResult r;
    r.mlirStep.ok   = true;
    r.llvmIrStep.ok = false;
    r.asmStep.ok    = true;
    CHECK(!r.allOk());
}

TEST(CodegenResult, NotOkIfAsmFailed) {
    nnc::CodegenResult r;
    r.mlirStep.ok   = true;
    r.llvmIrStep.ok = true;
    r.asmStep.ok    = false;
    CHECK(!r.allOk());
}

TEST(CodegenResult, AsmFileReturnsAsmOutput) {
    nnc::CodegenResult r;
    r.asmStep.outputFile = "/tmp/my_model.s";
    CHECK_EQ(r.asmFile(), std::string("/tmp/my_model.s"));
}

// ----------------------------------------------------------------
// Suite: EmitMLIR — шаг 1 пайплайна
// ----------------------------------------------------------------

TEST(EmitMLIR, CreatesOutputFile) {
    fs::path tmpDir = fs::temp_directory_path() / "nnc_test_emit";
    fs::create_directories(tmpDir);

    nnc::CodegenOptions opts;
    opts.workDir        = tmpDir;
    opts.outputBaseName = "unit_test_out";
    opts.verbose        = false;

    nnc::Codegen cg(opts);
    auto result = cg.emitMLIR(makeSimpleGraph());

    CHECK(result.ok);
    CHECK(!result.outputFile.empty());
    CHECK(fs::exists(result.outputFile));

    // Убираем за собой
    fs::remove_all(tmpDir);
}

TEST(EmitMLIR, OutputFileHasMlirExtension) {
    fs::path tmpDir = fs::temp_directory_path() / "nnc_test_ext";
    fs::create_directories(tmpDir);

    nnc::CodegenOptions opts;
    opts.workDir        = tmpDir;
    opts.outputBaseName = "my_model";

    nnc::Codegen cg(opts);
    auto result = cg.emitMLIR(makeSimpleGraph());

    std::string f = result.outputFile;
    CHECK(f.size() > 5 && f.substr(f.size()-5) == ".mlir");

    fs::remove_all(tmpDir);
}

TEST(EmitMLIR, OutputContainsValidMLIR) {
    fs::path tmpDir = fs::temp_directory_path() / "nnc_test_content";
    fs::create_directories(tmpDir);

    nnc::CodegenOptions opts;
    opts.workDir        = tmpDir;
    opts.outputBaseName = "content_test";

    nnc::Codegen cg(opts);
    auto result = cg.emitMLIR(makeSimpleGraph("relu_net"));

    CHECK(result.ok);

    std::ifstream f(result.outputFile);
    std::ostringstream ss; ss << f.rdbuf();
    std::string content = ss.str();

    CHECK_CONTAINS(content, "module {");
    CHECK_CONTAINS(content, "func.func @forward");
    CHECK_CONTAINS(content, "linalg.generic");   // Relu
    CHECK_CONTAINS(content, "return");

    fs::remove_all(tmpDir);
}

TEST(EmitMLIR, OutputBaseNameApplied) {
    fs::path tmpDir = fs::temp_directory_path() / "nnc_test_name";
    fs::create_directories(tmpDir);

    nnc::CodegenOptions opts;
    opts.workDir        = tmpDir;
    opts.outputBaseName = "custom_name_xyz";

    nnc::Codegen cg(opts);
    auto result = cg.emitMLIR(makeSimpleGraph());

    CHECK_CONTAINS(result.outputFile, "custom_name_xyz");

    fs::remove_all(tmpDir);
}

TEST(EmitMLIR, WorkDirCreatedAutomatically) {
    // Директория не существует — Codegen должен её создать
    fs::path tmpDir = fs::temp_directory_path() / "nnc_autodir_test" / "sub" / "deep";
    fs::remove_all(tmpDir.parent_path().parent_path());  // убедимся что нет

    nnc::CodegenOptions opts;
    opts.workDir        = tmpDir;
    opts.outputBaseName = "auto";

    nnc::Codegen cg(opts);
    CHECK(fs::exists(tmpDir));  // должна создаться в конструкторе

    fs::remove_all(tmpDir.parent_path().parent_path());
}

TEST(EmitMLIR, FuncNamePropagated) {
    fs::path tmpDir = fs::temp_directory_path() / "nnc_func_name";
    fs::create_directories(tmpDir);

    nnc::CodegenOptions opts;
    opts.workDir        = tmpDir;
    opts.outputBaseName = "fn_test";
    opts.funcName       = "my_special_forward";

    nnc::Codegen cg(opts);
    auto result = cg.emitMLIR(makeSimpleGraph());

    std::ifstream f(result.outputFile);
    std::ostringstream ss; ss << f.rdbuf();

    CHECK_CONTAINS(ss.str(), "@my_special_forward");
    CHECK_NOT_CONTAINS(ss.str(), "@forward");

    fs::remove_all(tmpDir);
}

TEST(EmitMLIR, MultipleCallsOverwrite) {
    // Второй вызов должен перезаписать файл
    fs::path tmpDir = fs::temp_directory_path() / "nnc_overwrite";
    fs::create_directories(tmpDir);

    nnc::CodegenOptions opts;
    opts.workDir        = tmpDir;
    opts.outputBaseName = "ow_test";

    nnc::Codegen cg(opts);

    auto g1 = makeSimpleGraph("net1");
    auto g2 = makeSimpleGraph("net2");

    auto r1 = cg.emitMLIR(g1);
    auto r2 = cg.emitMLIR(g2);

    CHECK(r1.ok);
    CHECK(r2.ok);
    CHECK_EQ(r1.outputFile, r2.outputFile);  // тот же файл

    // Файл содержит имя второго графа
    std::ifstream f(r2.outputFile);
    std::ostringstream ss; ss << f.rdbuf();
    CHECK_CONTAINS(ss.str(), "net2");

    fs::remove_all(tmpDir);
}

// ----------------------------------------------------------------
// Suite: CompileFromMLIR — поведение при отсутствии LLVM-инструментов
// ----------------------------------------------------------------

TEST(CompileFromMLIR, MissingFileFailsGracefully) {
    // Файл не существует → шаг должен вернуть !ok, а не упасть
    nnc::CodegenOptions opts;
    opts.workDir        = fs::temp_directory_path();
    opts.outputBaseName = "missing_test";
    opts.verbose        = false;

    nnc::Codegen cg(opts);
    // Несуществующий mlir-файл
    auto result = cg.compileFromMLIR("/tmp/does_not_exist_at_all.mlir");

    // Пайплайн должен вернуть результат (не крашнуться)
    // mlirStep помечается ok вручную в compileFromMLIR, поэтому проверяем llvmIrStep
    CHECK(!result.allOk());
}

TEST(CompileFromMLIR, WithBogusTools_LLVMStepFails) {
    // Указываем заведомо несуществующие инструменты
    fs::path tmpDir = fs::temp_directory_path() / "nnc_bogus_tools";
    fs::create_directories(tmpDir);

    nnc::CodegenOptions opts;
    opts.workDir            = tmpDir;
    opts.outputBaseName     = "bogus";
    opts.mlirOptPath        = "/no/such/mlir-opt";
    opts.mlirTranslatePath  = "/no/such/mlir-translate";
    opts.llcPath            = "/no/such/llc";
    opts.verbose            = false;
    opts.applyMlirPasses    = true;

    nnc::Codegen cg(opts);

    // Сначала создадим валидный .mlir
    auto g = makeSimpleGraph();
    auto mlirResult = cg.emitMLIR(g);
    CHECK(mlirResult.ok);

    // Затем попытаемся скомпилировать с несуществующим mlir-opt
    auto result = cg.compileFromMLIR(mlirResult.outputFile);

    CHECK(!result.llvmIrStep.ok);   // mlir-opt упал
    CHECK(!result.allOk());

    fs::remove_all(tmpDir);
}

// ----------------------------------------------------------------
// Suite: CodegenOptions — настройки пайплайна
// ----------------------------------------------------------------

TEST(CodegenOptionsUsage, VerboseFlagSetCorrectly) {
    nnc::CodegenOptions opts;
    opts.verbose = true;
    nnc::Codegen cg(opts);
    CHECK(cg.options().verbose);
}

TEST(CodegenOptionsUsage, OptLevelRange) {
    for (int lvl : {0, 1, 2, 3}) {
        nnc::CodegenOptions opts;
        opts.optLevel = lvl;
        nnc::Codegen cg(opts);
        CHECK_EQ(cg.options().optLevel, lvl);
    }
}

TEST(CodegenOptionsUsage, CustomToolPathsStored) {
    nnc::CodegenOptions opts;
    opts.mlirOptPath       = "/custom/mlir-opt-17";
    opts.mlirTranslatePath = "/custom/mlir-translate-17";
    opts.llcPath           = "/custom/llc-17";

    nnc::Codegen cg(opts);
    CHECK_EQ(cg.options().mlirOptPath,       std::string("/custom/mlir-opt-17"));
    CHECK_EQ(cg.options().mlirTranslatePath, std::string("/custom/mlir-translate-17"));
    CHECK_EQ(cg.options().llcPath,           std::string("/custom/llc-17"));
}

TEST(CodegenOptionsUsage, AllTargetArchsValid) {
    // Все архитектуры должны давать непустую строку (или "" для native)
    using A = nnc::TargetArch;
    CHECK_EQ(nnc::targetArchToString(A::Native),  std::string(""));
    CHECK_NE(nnc::targetArchToString(A::X86_64),  std::string(""));
    CHECK_NE(nnc::targetArchToString(A::AArch64), std::string(""));
    CHECK_NE(nnc::targetArchToString(A::RISCV64), std::string(""));
    CHECK_NE(nnc::targetArchToString(A::WASM32),  std::string(""));
}

// ----------------------------------------------------------------
// Suite: Integration — Graph → MLIR файл содержит все операции
// ----------------------------------------------------------------

TEST(Integration, FullGraphAllOps) {
    // Граф со всеми поддерживаемыми операциями
    fs::path tmpDir = fs::temp_directory_path() / "nnc_all_ops";
    fs::create_directories(tmpDir);

    nnc::Graph g; g.name = "all_ops";
    auto mkT = [](const std::string& n, std::vector<int64_t> d, bool init=false){
        nnc::TensorInfo t; t.name=n; t.dtype=nnc::DType::Float32;
        t.shape.dims=d; t.isInitializer=init; return t;
    };

    g.addTensor(mkT("x",     {1, 4}));
    g.addTensor(mkT("x2",    {1, 4}));
    g.addTensor(mkT("add_o", {1, 4}));
    g.addTensor(mkT("mul_o", {1, 4}));
    g.addTensor(mkT("rel_o", {1, 4}));
    g.addTensor(mkT("W",     {4, 2}, true));
    g.addTensor(mkT("mm_o",  {1, 2}));

    g.inputs  = {"x", "x2"};
    g.outputs = {"mm_o"};

    g.nodes.push_back({ "a0", nnc::OpType::Add,    {"x","x2"},     {"add_o"} });
    g.nodes.push_back({ "m0", nnc::OpType::Mul,    {"x","add_o"},  {"mul_o"} });
    g.nodes.push_back({ "r0", nnc::OpType::Relu,   {"mul_o"},      {"rel_o"} });
    g.nodes.push_back({ "mm", nnc::OpType::MatMul, {"rel_o","W"},  {"mm_o"}  });

    nnc::CodegenOptions opts;
    opts.workDir = tmpDir; opts.outputBaseName = "all_ops"; opts.verbose = false;

    nnc::Codegen cg(opts);
    auto result = cg.emitMLIR(g);

    CHECK(result.ok);
    std::ifstream f(result.outputFile);
    std::ostringstream ss; ss << f.rdbuf();
    std::string content = ss.str();

    CHECK_CONTAINS(content, "linalg.add");
    CHECK_CONTAINS(content, "linalg.mul");
    CHECK_CONTAINS(content, "linalg.generic");  // Relu
    CHECK_CONTAINS(content, "linalg.matmul");

    fs::remove_all(tmpDir);
}

TEST(Integration, GeneratedMLIRIsNonEmpty) {
    fs::path tmpDir = fs::temp_directory_path() / "nnc_nonempty";
    fs::create_directories(tmpDir);

    nnc::CodegenOptions opts;
    opts.workDir = tmpDir; opts.outputBaseName = "size_check";

    nnc::Codegen cg(opts);
    auto result = cg.emitMLIR(makeSimpleGraph());

    CHECK(result.ok);
    CHECK(fs::file_size(result.outputFile) > 100);  // хотя бы 100 байт

    fs::remove_all(tmpDir);
}
