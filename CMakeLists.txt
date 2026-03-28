cmake_minimum_required(VERSION 3.16)
project(nnc-codegen VERSION 0.1.0 LANGUAGES CXX)

# ============================================================
# CMakeLists.txt — Система сборки nnc-codegen
#
# Требования:
#   • CMake >= 3.16
#   • Компилятор с поддержкой C++17
#   • LLVM/MLIR >= 16 (опционально, для реального запуска пайплайна)
#
# Быстрый старт:
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   make -j$(nproc)
# ============================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Предупреждения компилятора
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wno-unused-parameter)
endif()

# ---- Библиотека nnc-core ----
# Включает граф и генераторы MLIR/Codegen.
# Выделяем в отдельную цель, чтобы тесты могли линковаться без main.
add_library(nnc-core
    src/graph.cpp
    src/mlir_emitter.cpp
    src/codegen.cpp
)
target_include_directories(nnc-core PUBLIC include)

# ---- Исполняемый файл драйвера ----
add_executable(nnc-codegen src/main.cpp)
target_link_libraries(nnc-codegen PRIVATE nnc-core)

# ---- Тесты ----
enable_testing()

add_executable(nnc-codegen-tests tests/test_mlir_emitter.cpp)
target_link_libraries(nnc-codegen-tests PRIVATE nnc-core)

add_test(NAME unit_tests COMMAND nnc-codegen-tests)

# ---- Установка (опционально) ----
install(TARGETS nnc-codegen DESTINATION bin)
install(DIRECTORY include/ DESTINATION include/nnc)

# ---- Вывод конфигурации ----
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
