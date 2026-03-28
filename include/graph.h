#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <variant>
#include <optional>


namespace nnc {

// Тип элементов тензора (подмножество типов ONNX)
enum class DType {
    Float32,
    Float64,
    Int32,
    Int64,
    Bool,
    Unknown
};

std::string dtypeToString(DType dt);

// Форма тензора: вектор размерностей (−1 = динамическая)
struct TensorShape {
    std::vector<int64_t> dims;

    bool isScalar()  const { return dims.empty(); }
    int64_t rank()   const { return static_cast<int64_t>(dims.size()); }
    std::string toString() const;
};

// Информация о тензоре (вход/выход операции или инициализатор)
struct TensorInfo {
    std::string  name;
    DType        dtype  = DType::Unknown;
    TensorShape  shape;
    bool         isInitializer = false;   // константный вес сети

    // Плоские данные инициализатора (если есть)
    std::vector<float> floatData;
};

// ---- Атрибуты операций ----

// Атрибут Conv: параметры свёртки
struct ConvAttrs {
    std::vector<int64_t> dilations;   // расширение ядра
    int64_t              group       = 1;
    std::vector<int64_t> kernelShape;
    std::string          autoPad     = "NOTSET";
    std::vector<int64_t> pads;
    std::vector<int64_t> strides;
};

// Атрибут Gemm: флаги транспонирования и масштаб
struct GemmAttrs {
    float   alpha   = 1.0f;
    float   beta    = 1.0f;
    int64_t transA  = 0;
    int64_t transB  = 0;
};

// Атрибут Reshape / операций с аргументом axis
struct AxisAttr {
    int64_t axis = 0;
};

// Объединение всех возможных атрибутов
using OpAttrs = std::variant<
    std::monostate,   // нет атрибутов
    ConvAttrs,
    GemmAttrs,
    AxisAttr
>;

// ---- Типы поддерживаемых операций ----
enum class OpType {
    Add,
    Mul,
    Conv,
    Relu,
    MatMul,
    Gemm,
    // Расширяйте список по мере необходимости
    Unknown
};

std::string opTypeToString(OpType op);
OpType      opTypeFromString(const std::string& s);

// ---- Узел вычислительного графа (операция) ----
struct Node {
    std::string            name;     // уникальное имя узла
    OpType                 opType;
    std::vector<std::string> inputs;  // имена входных тензоров
    std::vector<std::string> outputs; // имена выходных тензоров
    OpAttrs                attrs;    // атрибуты, специфичные для операции
};

// ---- Вычислительный граф ----
class Graph {
public:
    std::string name;

    // Узлы графа в топологическом порядке
    std::vector<Node> nodes;

    // Все тензоры: входы сети, выходы, инициализаторы, промежуточные
    std::unordered_map<std::string, TensorInfo> tensors;

    // Входы и выходы всего графа
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;

    // Вспомогательные методы
    const TensorInfo* getTensor(const std::string& name) const;
    void              addTensor(TensorInfo t);
    void              dump(bool verbose = false) const;  // отладочный вывод
};

}
