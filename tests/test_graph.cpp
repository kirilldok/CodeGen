#include "../include/test_framework.h"
#include "../include/graph.h"
#include <sstream>

// Suite: DType

TEST(DType, ToStringKnownTypes) {
    CHECK_EQ(nnc::dtypeToString(nnc::DType::Float32), std::string("float32"));
    CHECK_EQ(nnc::dtypeToString(nnc::DType::Float64), std::string("float64"));
    CHECK_EQ(nnc::dtypeToString(nnc::DType::Int32),   std::string("int32"));
    CHECK_EQ(nnc::dtypeToString(nnc::DType::Int64),   std::string("int64"));
    CHECK_EQ(nnc::dtypeToString(nnc::DType::Bool),    std::string("bool"));
}

TEST(DType, ToStringUnknown) {
    CHECK_EQ(nnc::dtypeToString(nnc::DType::Unknown), std::string("unknown"));
}


// Suite: TensorShape

TEST(TensorShape, ScalarIsEmpty) {
    nnc::TensorShape s;
    CHECK(s.isScalar());
    CHECK_EQ(s.rank(), int64_t(0));
}

TEST(TensorShape, RankMatchesDims) {
    nnc::TensorShape s;
    s.dims = {1, 3, 224, 224};
    CHECK_EQ(s.rank(), int64_t(4));
    CHECK(!s.isScalar());
}

TEST(TensorShape, ToStringStatic) {
    nnc::TensorShape s;
    s.dims = {1, 3, 224, 224};
    CHECK_EQ(s.toString(), std::string("[1, 3, 224, 224]"));
}

TEST(TensorShape, ToStringDynamic) {
    nnc::TensorShape s;
    s.dims = {-1, 128};
    CHECK_EQ(s.toString(), std::string("[?, 128]"));
}

TEST(TensorShape, ToStringScalar) {
    nnc::TensorShape s;
    CHECK_EQ(s.toString(), std::string("scalar"));
}

TEST(TensorShape, SingleDim) {
    nnc::TensorShape s;
    s.dims = {512};
    CHECK_EQ(s.rank(), int64_t(1));
    CHECK_EQ(s.toString(), std::string("[512]"));
}

TEST(TensorShape, AllDynamic) {
    nnc::TensorShape s;
    s.dims = {-1, -1, -1};
    CHECK_EQ(s.toString(), std::string("[?, ?, ?]"));
}


// Suite: TensorInfo

TEST(TensorInfo, DefaultNotInitializer) {
    nnc::TensorInfo t;
    t.name  = "x";
    t.dtype = nnc::DType::Float32;
    t.shape.dims = {4, 4};
    CHECK(!t.isInitializer);
    CHECK(t.floatData.empty());
}

TEST(TensorInfo, InitializerFlag) {
    nnc::TensorInfo t;
    t.name          = "weight";
    t.isInitializer = true;
    t.floatData     = {0.1f, 0.2f, 0.3f};
    CHECK(t.isInitializer);
    CHECK_EQ(t.floatData.size(), size_t(3));
}


// Suite: OpType

TEST(OpType, RoundTripAllSupported) {
    using namespace nnc;
    for (auto op : {OpType::Add, OpType::Mul, OpType::Conv,
                    OpType::Relu, OpType::MatMul, OpType::Gemm}) {
        CHECK_EQ(opTypeFromString(opTypeToString(op)), op);
    }
}

TEST(OpType, UnknownString) {
    CHECK_EQ(nnc::opTypeFromString("NonExistentOp"), nnc::OpType::Unknown);
    CHECK_EQ(nnc::opTypeFromString(""),              nnc::OpType::Unknown);
}

TEST(OpType, CaseSensitive) {
    // "add" != "Add" - ONNX имена с заглавной буквы
    CHECK_EQ(nnc::opTypeFromString("add"), nnc::OpType::Unknown);
    CHECK_EQ(nnc::opTypeFromString("ADD"), nnc::OpType::Unknown);
}

TEST(OpType, UnknownToString) {
    CHECK_EQ(nnc::opTypeToString(nnc::OpType::Unknown), std::string("Unknown"));
}


// Suite: Graph

TEST(Graph, EmptyGraphHasNoTensors) {
    nnc::Graph g;
    CHECK(g.nodes.empty());
    CHECK(g.inputs.empty());
    CHECK(g.outputs.empty());
    CHECK(g.tensors.empty());
}

TEST(Graph, AddAndGetTensor) {
    nnc::Graph g;
    nnc::TensorInfo t;
    t.name  = "data";
    t.dtype = nnc::DType::Float32;
    t.shape.dims = {1, 64};
    g.addTensor(t);

    const nnc::TensorInfo* found = g.getTensor("data");
    CHECK(found != nullptr);
    CHECK_EQ(found->name, std::string("data"));
    CHECK_EQ(found->dtype, nnc::DType::Float32);
    CHECK_EQ(found->shape.rank(), int64_t(2));
}

TEST(Graph, GetMissingTensorReturnsNull) {
    nnc::Graph g;
    CHECK(g.getTensor("does_not_exist") == nullptr);
}

TEST(Graph, OverwriteTensorWithSameName) {
    nnc::Graph g;

    nnc::TensorInfo t1; t1.name = "x"; t1.dtype = nnc::DType::Float32;
    nnc::TensorInfo t2; t2.name = "x"; t2.dtype = nnc::DType::Int32;

    g.addTensor(t1);
    g.addTensor(t2);  // перезаписать

    CHECK_EQ(g.getTensor("x")->dtype, nnc::DType::Int32);
}

TEST(Graph, MultipleInputsOutputs) {
    nnc::Graph g;
    g.inputs  = {"a", "b", "c"};
    g.outputs = {"out1", "out2"};
    CHECK_EQ(g.inputs.size(),  size_t(3));
    CHECK_EQ(g.outputs.size(), size_t(2));
}

TEST(Graph, NodeOrdering) {
    // Порядок узлов сохраняется (топологический порядок)
    nnc::Graph g;
    for (int i = 0; i < 5; ++i) {
        nnc::Node n;
        n.name   = "node_" + std::to_string(i);
        n.opType = nnc::OpType::Relu;
        g.nodes.push_back(n);
    }
    CHECK_EQ(g.nodes.size(), size_t(5));
    CHECK_EQ(g.nodes[0].name, std::string("node_0"));
    CHECK_EQ(g.nodes[4].name, std::string("node_4"));
}

TEST(Graph, DumpDoesNotCrash) {
    // smoke-тест: dump не должен кидать исключений
    nnc::Graph g;
    g.name = "smoke";
    g.inputs = {"in"};
    g.outputs = {"out"};

    nnc::Node n; n.name = "relu_0"; n.opType = nnc::OpType::Relu;
    n.inputs = {"in"}; n.outputs = {"out"};
    g.nodes.push_back(n);

    // перенаправляем cout, чтобы не засорять тестовый вывод
    std::ostringstream buf;
    std::streambuf* old = std::cout.rdbuf(buf.rdbuf());
    CHECK_NO_THROW(g.dump(true));
    std::cout.rdbuf(old);

    CHECK_CONTAINS(buf.str(), "smoke");
    CHECK_CONTAINS(buf.str(), "relu_0");
}


// Suite: ConvAttrs

TEST(ConvAttrs, DefaultValues) {
    nnc::ConvAttrs ca;
    CHECK_EQ(ca.group,    int64_t(1));
    CHECK_EQ(ca.autoPad,  std::string("NOTSET"));
    CHECK(ca.dilations.empty());
    CHECK(ca.strides.empty());
    CHECK(ca.pads.empty());
    CHECK(ca.kernelShape.empty());
}


// Suite: GemmAttrs

TEST(GemmAttrs, DefaultValues) {
    nnc::GemmAttrs ga;
    CHECK_EQ(ga.alpha,  1.0f);
    CHECK_EQ(ga.beta,   1.0f);
    CHECK_EQ(ga.transA, int64_t(0));
    CHECK_EQ(ga.transB, int64_t(0));
}
