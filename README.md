Кодогенератор для нейронных сетей: преобразует вычислительный граф (построенный в задании FE из ONNX) в MLIR, затем через LLVM в ассемблер целевой архитектуры.

---

## Архитектура пайплайна

```
Graph (ONNX / In-Memory)
       │
       ▼
  MLIREmitter            ← src/mlir_emitter.cpp
  (func + linalg диалекты)
       │  .mlir
       ▼
  mlir-opt               ← внешний инструмент LLVM
  (lowering passes: linalg→loops→SCF→CF→LLVM)
       │  _lowered.mlir
       ▼
  mlir-translate         ← внешний инструмент LLVM
  (--mlir-to-llvmir)
       │  .ll
       ▼
  llc                    ← внешний инструмент LLVM
  (--march=<arch> -O2)
       │  .s
       ▼
  Assembly (x86-64 / AArch64 / RISC-V / WASM)
```

---

## Структура директорий

```
nnc-codegen/
├── CMakeLists.txt          # система сборки
├── README.md               # этот файл
├── include/
│   ├── graph.h             # структура данных графа (из задания FE)
│   ├── mlir_emitter.h      # генератор MLIR
│   └── codegen.h           # фасад пайплайна, опции командной строки
├── src/
│   ├── graph.cpp
│   ├── mlir_emitter.cpp    # реализация генерации MLIR
│   ├── codegen.cpp         # вызов mlir-opt / mlir-translate / llc
│   └── main.cpp            # драйвер с разбором аргументов CLI
├── tests/
│   └── test_mlir_emitter.cpp  # юнит-тесты (без LLVM)
├── examples/
│   └── simple_mlp.mlir     # готовый MLIR для тестирования пайплайна
└── scripts/
    └── verify.py           # верификация через PyTorch / ONNXRuntime
```

---

## Требования

### Обязательно
- **CMake** ≥ 3.16
- **C++17**-совместимый компилятор (GCC ≥ 9, Clang ≥ 10, MSVC 2019)

### Для полного пайплайна (Graph → Assembly)
- **LLVM/MLIR** ≥ 16 с инструментами:
  - `mlir-opt`
  - `mlir-translate`
  - `llc`

#### Установка LLVM (Ubuntu/Debian)
```bash
# Скрипт установки от LLVM
bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)" -- 18
sudo apt-get install -y mlir-18-tools llvm-18-tools

# Добавить в PATH (или передавать пути флагами)
export PATH="/usr/lib/llvm-18/bin:$PATH"
```

#### Установка LLVM (macOS)
```bash
brew install llvm@18
export PATH="$(brew --prefix llvm@18)/bin:$PATH"
```

#### Сборка LLVM из исходников (если пакеты не подходят)
```bash
git clone https://github.com/llvm/llvm-project --depth=1
cd llvm-project
cmake -S llvm -B build \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="X86;AArch64;RISCV" \
    -DCMAKE_BUILD_TYPE=Release \
    -G Ninja
cmake --build build -j$(nproc)
export PATH="$PWD/build/bin:$PATH"
```

---

## Сборка проекта

```bash
# 1. Клонировать репозиторий
git clone <repo-url>
cd nnc-codegen

# 2. Создать директорию сборки
mkdir build && cd build

# 3. Конфигурация
cmake .. -DCMAKE_BUILD_TYPE=Release

# Для Debug-сборки:
# cmake .. -DCMAKE_BUILD_TYPE=Debug

# 4. Компиляция
make -j$(nproc)
# или: cmake --build . -j$(nproc)
```

После сборки в `build/` появятся:
- `nnc-codegen` — основной исполняемый файл
- `nnc-codegen-tests` — юнит-тесты

---

## Запуск

### Справка

```bash
./nnc-codegen --help
```

### Демо-режим (без ONNX, тестовый граф)

Запускает встроенный граф `Add → ReLU → MatMul` и компилирует его:

```bash
./nnc-codegen --verbose
```

Результаты сохраняются в текущую директорию:
- `output.mlir` — сгенерированный MLIR
- `output_lowered.mlir` — после passes mlir-opt
- `output.ll` — LLVM IR
- `output.s` — ассемблер

### Компиляция готового .mlir файла

```bash
./nnc-codegen ../examples/simple_mlp.mlir \
    --out simple_mlp \
    --outdir /tmp/nnc_out \
    --verbose
```

### Кросс-компиляция в AArch64

```bash
./nnc-codegen ../examples/simple_mlp.mlir \
    --arch aarch64 \
    --out model_arm \
    --outdir /tmp
```

### Использование кастомного LLVM

```bash
./nnc-codegen ../examples/simple_mlp.mlir \
    --mlir-opt   /opt/llvm-18/bin/mlir-opt \
    --mlir-trans /opt/llvm-18/bin/mlir-translate \
    --llc        /opt/llvm-18/bin/llc \
    --verbose
```

### Опции командной строки

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--mlir-opt <path>` | `mlir-opt` | Путь к mlir-opt |
| `--mlir-trans <path>` | `mlir-translate` | Путь к mlir-translate |
| `--llc <path>` | `llc` | Путь к llc |
| `--arch <arch>` | `native` | Архитектура: `native`, `x86_64`, `aarch64`, `riscv64`, `wasm32` |
| `-O<n>` | `-O2` | Уровень оптимизации (0–3) |
| `--out <name>` | `output` | Базовое имя выходных файлов |
| `--outdir <dir>` | `.` | Рабочая директория |
| `--no-passes` | — | Пропустить passes mlir-opt (только emit MLIR) |
| `--no-intermediate` | — | Удалять промежуточные `.ll` и `_lowered.mlir` |
| `--func <name>` | `forward` | Имя функции в MLIR/LLVM |
| `-v`, `--verbose` | — | Выводить запускаемые команды |
| `-h`, `--help` | — | Справка |

---

## Тесты

### Юнит-тесты (без LLVM)

Тесты проверяют корректность генерации MLIR без реального запуска пайплайна:

```bash
cd build
./nnc-codegen-tests
```

или через CTest:

```bash
cd build
ctest --output-on-failure
```

### Верификация через PyTorch / ONNXRuntime

```bash
pip install torch onnx onnxruntime numpy
python3 scripts/verify.py
```

Скрипт:
1. Создаёт модель `Linear → ReLU → Linear` в PyTorch
2. Экспортирует в ONNX
3. Запускает через ONNXRuntime (эталон)
4. Генерирует MLIR через `nnc-codegen` и показывает результат

---

## Поддерживаемые операции

| ONNX Op | MLIR диалект | Статус |
|---------|-------------|--------|
| `Add`   | `linalg.add` | ✅ |
| `Mul`   | `linalg.mul` | ✅ |
| `Conv`  | `linalg.conv_2d_nchw_fchw` | ✅ |
| `Relu`  | `linalg.generic` + `arith.maxf` | ✅ |
| `MatMul`| `linalg.matmul` | ✅ |
| `Gemm`  | `linalg.matmul` + `linalg.add` | ✅ |

---

## Проход lowering (mlir-opt)

```
linalg-bufferize          # тензоры → memref буферы
convert-linalg-to-loops   # linalg ops → SCF циклы
convert-scf-to-cf         # SCF → Control Flow
convert-cf-to-llvm         # CF → LLVM диалект
convert-arith-to-llvm      # arith → LLVM
convert-func-to-llvm       # func → llvm.func
reconcile-unrealized-casts # уборка лишних приведений типов
canonicalize               # CSE + DCE + упрощение
```

---

## Пример вывода

Запуск `./nnc-codegen --verbose` (демо-граф Add → Relu → MatMul):

```
Graph: demo_graph
Inputs:  input
Outputs: output
Nodes (3):
  [0] Add 'add_0'
  [1] Relu 'relu_0'
  [2] MatMul 'matmul_0'

[nnc-codegen] $ mlir-opt --linalg-bufferize ... output.mlir -o output_lowered.mlir
[nnc-codegen] $ mlir-translate --mlir-to-llvmir output_lowered.mlir -o output.ll
[nnc-codegen] $ llc -O2 -filetype=asm output.ll -o output.s

=== Compilation results ===
  MLIR emission:  OK  →  output.mlir
  LLVM IR       : OK  →  output.ll
  Assembly      : OK  →  output.s

Success! Assembly: output.s
```

---

## Связь с заданием FE

В задании FE был реализован парсер ONNX → `nnc::Graph`.  
Данный модуль принимает `nnc::Graph` на вход и транслирует его в ассемблер.  
Интеграция выглядит так:

```cpp
// Парсинг ONNX (из задания FE)
nnc::Graph graph = nnc::OnnxParser::parse("model.onnx");

// Кодогенерация (это задание)
nnc::CodegenOptions opts;
opts.verbose = true;
opts.targetArch = nnc::TargetArch::Native;

nnc::Codegen codegen(opts);
auto result = codegen.run(graph);

if (result.allOk())
    std::cout << "Assembly: " << result.asmFile() << "\n";
```

---

## Авторы

Курс «Введение в тензорные компиляторы»  
Преподаватели: @synthmoza, @vloznovenko
