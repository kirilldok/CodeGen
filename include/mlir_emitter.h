#pragma once

#include "../include/../include/graph.h"
#include <ostream>
#include <string>

// ============================================================
// ../include/../include/mlir_emitter.h — Генератор MLIR-представления
//
// Преобразует вычислительный граф (nnc::Graph) в текстовый
// MLIR-код с использованием диалектов:
//   • func  — функции и вызовы
//   • arith — арифметические примитивы
//   • linalg — тензорные операции (conv, matmul, …)
//   • tensor — абстракция над многомерными массивами
//
// Полученный .mlir-файл затем обрабатывается цепочкой:
//   mlir-opt → mlir-translate → llc
// ============================================================

namespace nnc {

// Параметры генерации MLIR
struct MLIREmitOptions {
    // Добавлять ли комментарии с именами узлов ONNX
    bool emitNodeComments = true;
    // Оборачивать функцию в module {}
    bool emitModuleWrapper = true;
    // Имя генерируемой MLIR-функции
    std::string funcName = "forward";
};

class MLIREmitter {
public:
    explicit MLIREmitter(const MLIREmitOptions& opts = {});

    // Основной метод: записывает MLIR-текст в поток os
    void emit(const Graph& graph, std::ostream& os) const;

    // Удобная перегрузка: возвращает строку
    std::string emitToString(const Graph& graph) const;

private:
    MLIREmitOptions opts_;

    // Вспомогательные генераторы для каждого типа операции
    void emitAdd   (const Node& node, const Graph& g, std::ostream& os, int indent) const;
    void emitMul   (const Node& node, const Graph& g, std::ostream& os, int indent) const;
    void emitConv  (const Node& node, const Graph& g, std::ostream& os, int indent) const;
    void emitRelu  (const Node& node, const Graph& g, std::ostream& os, int indent) const;
    void emitMatMul(const Node& node, const Graph& g, std::ostream& os, int indent) const;
    void emitGemm  (const Node& node, const Graph& g, std::ostream& os, int indent) const;

    // Преобразование типа тензора в MLIR-тип (например, "f32")
    static std::string mlirScalarType(DType dt);
    // Преобразование формы тензора в MLIR memref/tensor-тип
    // Например: [1, 3, 224, 224] × f32 → "tensor<1x3x224x224xf32>"
    static std::string mlirTensorType(const TensorInfo& t);
    // SSA-имя для тензора (добавляем '%' к имени)
    static std::string ssaName(const std::string& name);
    // Отступ
    static std::string ind(int n);
};

} // namespace nnc
