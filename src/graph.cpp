#include "../include/../include/graph.h"
#include <iostream>
#include <sstream>



namespace nnc {

// ---- DType -> строка ----
std::string dtypeToString(DType dt) {
    switch (dt) {
        case DType::Float32: return "float32";
        case DType::Float64: return "float64";
        case DType::Int32:   return "int32";
        case DType::Int64:   return "int64";
        case DType::Bool:    return "bool";
        default:             return "unknown";
    }
}

// TensorShape
std::string TensorShape::toString() const {
    if (dims.empty()) return "scalar";
    std::string s = "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i) s += ", ";
        s += (dims[i] < 0 ? "?" : std::to_string(dims[i]));
    }
    return s + "]";
}

// ---- OpType <-> строка ----
std::string opTypeToString(OpType op) {
    switch (op) {
        case OpType::Add:    return "Add";
        case OpType::Mul:    return "Mul";
        case OpType::Conv:   return "Conv";
        case OpType::Relu:   return "Relu";
        case OpType::MatMul: return "MatMul";
        case OpType::Gemm:   return "Gemm";
        default:             return "Unknown";
    }
}

OpType opTypeFromString(const std::string& s) {
    if (s == "Add")    return OpType::Add;
    if (s == "Mul")    return OpType::Mul;
    if (s == "Conv")   return OpType::Conv;
    if (s == "Relu")   return OpType::Relu;
    if (s == "MatMul") return OpType::MatMul;
    if (s == "Gemm")   return OpType::Gemm;
    return OpType::Unknown;
}

// ---- Graph ----
const TensorInfo* Graph::getTensor(const std::string& name) const {
    auto it = tensors.find(name);
    return (it != tensors.end()) ? &it->second : nullptr;
}

void Graph::addTensor(TensorInfo t) {
    tensors[t.name] = std::move(t);
}

void Graph::dump(bool verbose) const {
    std::cout << "=== Graph: " << name << " ===\n";
    std::cout << "Inputs:  ";
    for (const auto& inp : inputs) std::cout << inp << " ";
    std::cout << "\nOutputs: ";
    for (const auto& out : outputs) std::cout << out << " ";
    std::cout << "\n\nNodes (" << nodes.size() << "):\n";

    for (size_t i = 0; i < nodes.size(); ++i) {
        const auto& n = nodes[i];
        std::cout << "  [" << i << "] " << opTypeToString(n.opType)
                  << " '" << n.name << "'\n";
        if (verbose) {
            std::cout << "       in : ";
            for (const auto& inp : n.inputs) std::cout << inp << " ";
            std::cout << "\n       out: ";
            for (const auto& out : n.outputs) std::cout << out << " ";
            std::cout << "\n";
        }
    }
}

} // namespace nnc
