#include "../include/../include/graph.h"
#include "../include/../include/mlir_emitter.h"
#include "../include/../include/codegen.h"
#include <cassert>
#include <iostream>
#include <sstream>
#include <string>


static int g_passed = 0;
static int g_failed = 0;

#define CHECK(cond, msg)                                          \
    do {                                                          \
        if (!(cond)) {                                            \
            std::cerr << "FAIL [" << __LINE__ << "]: " << msg << "\n"; \
            ++g_failed;                                           \
        } else {                                                  \
            ++g_passed;                                           \
        }                                                         \
    } while(0)

#define CHECK_CONTAINS(str, sub)                                         \
    CHECK((str).find(sub) != std::string::npos,                          \
          "Expected to find: \"" << sub << "\"")

// ---- Вспомогательные функции ----

static nnc::TensorInfo makeTensor(const std::string& name,
                                  nnc::DType dt,
                                  std::vector<int64_t> dims,
                                  bool isInit = false) {
    nnc::TensorInfo t;
    t.name  = name;
    t.dtype = dt;
    t.shape.dims = std::move(dims);
    t.isInitializer = isInit;
    return t;
}

// ---- Тест 1: TensorShape::toString ----
static void testTensorShape() {
    std::cout << "Test: TensorShape::toString ... ";
    nnc::TensorShape s;
    s.dims = {1, 3, 224, 224};
    CHECK(s.toString() == "[1, 3, 224, 224]", "static shape");

    nnc::TensorShape dyn;
    dyn.dims = {-1, 128};
    CHECK(dyn.toString() == "[?, 128]", "dynamic shape");

    nnc::TensorShape scalar;
    CHECK(scalar.isScalar(), "scalar");
    std::cout << "done\n";
}

// ---- Тест 2: opTypeFromString round-trip ----
static void testOpTypeRoundTrip() {
    std::cout << "Test: opType round-trip ... ";
    using namespace nnc;
    for (auto op : {OpType::Add, OpType::Mul, OpType::Conv,
                    OpType::Relu, OpType::MatMul, OpType::Gemm}) {
        CHECK(opTypeFromString(opTypeToString(op)) == op,
              "round-trip for " + opTypeToString(op));
    }
    std::cout << "done\n";
}

// ---- Тест 3: генерация Add ----
static void testEmitAdd() {
    std::cout << "Test: emit Add ... ";
    nnc::Graph g;
    g.name = "test_add";
    g.addTensor(makeTensor("x",   nnc::DType::Float32, {4, 4}));
    g.addTensor(makeTensor("y",   nnc::DType::Float32, {4, 4}));
    g.addTensor(makeTensor("out", nnc::DType::Float32, {4, 4}));
    g.inputs  = {"x", "y"};
    g.outputs = {"out"};

    nnc::Node addNode;
    addNode.name   = "add_0";
    addNode.opType = nnc::OpType::Add;
    addNode.inputs  = {"x", "y"};
    addNode.outputs = {"out"};
    g.nodes.push_back(addNode);

    nnc::MLIREmitter emitter;
    std::string mlir = emitter.emitToString(g);

    CHECK_CONTAINS(mlir, "linalg.add");
    CHECK_CONTAINS(mlir, "%x");
    CHECK_CONTAINS(mlir, "%y");
    CHECK_CONTAINS(mlir, "tensor<4x4xf32>");
    std::cout << "done\n";
}

// ---- Тест 4: генерация Relu ----
static void testEmitRelu() {
    std::cout << "Test: emit Relu ... ";
    nnc::Graph g;
    g.name = "test_relu";
    g.addTensor(makeTensor("input",  nnc::DType::Float32, {1, 32}));
    g.addTensor(makeTensor("output", nnc::DType::Float32, {1, 32}));
    g.inputs  = {"input"};
    g.outputs = {"output"};

    nnc::Node relu;
    relu.name   = "relu_0";
    relu.opType = nnc::OpType::Relu;
    relu.inputs  = {"input"};
    relu.outputs = {"output"};
    g.nodes.push_back(relu);

    nnc::MLIREmitter emitter;
    std::string mlir = emitter.emitToString(g);

    CHECK_CONTAINS(mlir, "linalg.generic");
    CHECK_CONTAINS(mlir, "arith.maxf");
    CHECK_CONTAINS(mlir, "0.0");
    std::cout << "done\n";
}

// ---- Тест 5: генерация MatMul ----
static void testEmitMatMul() {
    std::cout << "Test: emit MatMul ... ";
    nnc::Graph g;
    g.name = "test_matmul";
    g.addTensor(makeTensor("A",   nnc::DType::Float32, {8, 16}));
    g.addTensor(makeTensor("B",   nnc::DType::Float32, {16, 4}));
    g.addTensor(makeTensor("C",   nnc::DType::Float32, {8, 4}));
    g.inputs  = {"A", "B"};
    g.outputs = {"C"};

    nnc::Node mm;
    mm.name   = "mm_0";
    mm.opType = nnc::OpType::MatMul;
    mm.inputs  = {"A", "B"};
    mm.outputs = {"C"};
    g.nodes.push_back(mm);

    nnc::MLIREmitter emitter;
    std::string mlir = emitter.emitToString(g);

    CHECK_CONTAINS(mlir, "linalg.matmul");
    CHECK_CONTAINS(mlir, "tensor<8x16xf32>");
    CHECK_CONTAINS(mlir, "tensor<16x4xf32>");
    CHECK_CONTAINS(mlir, "tensor<8x4xf32>");
    std::cout << "done\n";
}

// ---- Тест 6: генерация Conv2D ----
static void testEmitConv() {
    std::cout << "Test: emit Conv ... ";
    nnc::Graph g;
    g.name = "test_conv";
    g.addTensor(makeTensor("img",    nnc::DType::Float32, {1, 3, 28, 28}));
    g.addTensor(makeTensor("kernel", nnc::DType::Float32, {16, 3, 3, 3}, /*isInit=*/true));
    g.addTensor(makeTensor("feat",   nnc::DType::Float32, {1, 16, 26, 26}));
    g.inputs  = {"img"};
    g.outputs = {"feat"};

    nnc::ConvAttrs ca;
    ca.strides   = {1, 1};
    ca.dilations = {1, 1};
    ca.pads      = {0, 0, 0, 0};

    nnc::Node conv;
    conv.name   = "conv_0";
    conv.opType = nnc::OpType::Conv;
    conv.inputs  = {"img", "kernel"};
    conv.outputs = {"feat"};
    conv.attrs   = ca;
    g.nodes.push_back(conv);

    nnc::MLIREmitter emitter;
    std::string mlir = emitter.emitToString(g);

    CHECK_CONTAINS(mlir, "linalg.conv_2d_nchw_fchw");
    CHECK_CONTAINS(mlir, "dilations");
    CHECK_CONTAINS(mlir, "strides");
    std::cout << "done\n";
}

// ---- Тест 7: генерация Gemm с bias ----
static void testEmitGemm() {
    std::cout << "Test: emit Gemm ... ";
    nnc::Graph g;
    g.name = "test_gemm";
    g.addTensor(makeTensor("X",    nnc::DType::Float32, {4, 8}));
    g.addTensor(makeTensor("W",    nnc::DType::Float32, {8, 2}, /*isInit=*/true));
    g.addTensor(makeTensor("bias", nnc::DType::Float32, {4, 2}, /*isInit=*/true));
    g.addTensor(makeTensor("Y",    nnc::DType::Float32, {4, 2}));
    g.inputs  = {"X"};
    g.outputs = {"Y"};

    nnc::GemmAttrs ga;
    ga.alpha  = 1.0f;
    ga.beta   = 1.0f;
    ga.transA = 0;
    ga.transB = 0;

    nnc::Node gemm;
    gemm.name   = "gemm_0";
    gemm.opType = nnc::OpType::Gemm;
    gemm.inputs  = {"X", "W", "bias"};
    gemm.outputs = {"Y"};
    gemm.attrs   = ga;
    g.nodes.push_back(gemm);

    nnc::MLIREmitter emitter;
    std::string mlir = emitter.emitToString(g);

    CHECK_CONTAINS(mlir, "linalg.matmul");
    CHECK_CONTAINS(mlir, "linalg.add");
    CHECK_CONTAINS(mlir, "alpha=1");
    std::cout << "done\n";
}

// ---- Тест 8: полная структура модуля ----
static void testModuleWrapper() {
    std::cout << "Test: module wrapper ... ";
    nnc::Graph g;
    g.name = "tiny";
    g.addTensor(makeTensor("in",  nnc::DType::Float32, {2, 2}));
    g.addTensor(makeTensor("out", nnc::DType::Float32, {2, 2}));
    g.inputs  = {"in"};
    g.outputs = {"out"};

    nnc::Node relu;
    relu.name   = "r";
    relu.opType = nnc::OpType::Relu;
    relu.inputs  = {"in"};
    relu.outputs = {"out"};
    g.nodes.push_back(relu);

    nnc::MLIREmitOptions opts;
    opts.funcName          = "my_forward";
    opts.emitModuleWrapper = true;

    nnc::MLIREmitter emitter(opts);
    std::string mlir = emitter.emitToString(g);

    CHECK_CONTAINS(mlir, "module {");
    CHECK_CONTAINS(mlir, "func.func @my_forward");
    CHECK_CONTAINS(mlir, "return");
    CHECK_CONTAINS(mlir, "} // module");
    std::cout << "done\n";
}

// ---- Тест 9: CodegenOptions - targetArchToString ----
static void testTargetArch() {
    std::cout << "Test: targetArchToString ... ";
    CHECK(nnc::targetArchToString(nnc::TargetArch::Native)  == "", "native");
    CHECK(nnc::targetArchToString(nnc::TargetArch::X86_64)  == "x86-64", "x86_64");
    CHECK(nnc::targetArchToString(nnc::TargetArch::AArch64) == "aarch64", "aarch64");
    CHECK(nnc::targetArchToString(nnc::TargetArch::RISCV64) == "riscv64", "riscv64");
    std::cout << "done\n";
}


int main() {
    std::cout << "=== nnc-codegen unit tests ===\n\n";

    testTensorShape();
    testOpTypeRoundTrip();
    testEmitAdd();
    testEmitRelu();
    testEmitMatMul();
    testEmitConv();
    testEmitGemm();
    testModuleWrapper();
    testTargetArch();

    std::cout << "\n=== Results: "
              << g_passed << " passed, "
              << g_failed << " failed ===\n";
    return g_failed == 0 ? 0 : 1;
}
