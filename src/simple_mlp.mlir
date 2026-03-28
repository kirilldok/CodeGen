// Пример вручную написанного MLIR для тестирования пайплайна.
//
// Этот файл можно скормить напрямую:
//   nnc-codegen examples/simple_mlp.mlir --out simple_mlp --verbose
//
// Описывает двухслойный персептрон:
//   relu(matmul(x, w1) + b1) → matmul(h, w2) + b2

module {

  // Вспомогательная функция: affine bias add
  func.func private @bias_add(%x: tensor<?x?xf32>, %b: tensor<?xf32>)
      -> tensor<?x?xf32> {
    // В реальном коде здесь был бы linalg.broadcast + linalg.add
    // Для демонстрации упрощаем до прямого return
    return %x : tensor<?x?xf32>
  }

  // Основная функция: forward pass простой MLP
  func.func @forward(%input : tensor<1x16xf32>,
                     %w1    : tensor<16x8xf32>,
                     %b1    : tensor<1x8xf32>,
                     %w2    : tensor<8x4xf32>,
                     %b2    : tensor<1x4xf32>)
      -> tensor<1x4xf32> {

    // Инициализаторы (нулевые буферы для накопления результата)
    %zero_h  = arith.constant dense<0.0> : tensor<1x8xf32>
    %zero_y  = arith.constant dense<0.0> : tensor<1x4xf32>

    // Слой 1: h_raw = input × w1
    %h_raw = linalg.matmul
        ins(%input, %w1 : tensor<1x16xf32>, tensor<16x8xf32>)
        outs(%zero_h : tensor<1x8xf32>)
        -> tensor<1x8xf32>

    // Добавляем bias b1 (упрощённо — linalg.add с одинаковым типом)
    %h_biased = linalg.add
        ins(%h_raw, %b1 : tensor<1x8xf32>, tensor<1x8xf32>)
        outs(%zero_h : tensor<1x8xf32>)
        -> tensor<1x8xf32>

    // Активация ReLU: h = max(h_biased, 0)
    %h = linalg.generic {
        indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
    }
    ins(%h_biased : tensor<1x8xf32>)
    outs(%zero_h  : tensor<1x8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %zero_f = arith.constant 0.0 : f32
        %relu   = arith.maxf %in, %zero_f : f32
        linalg.yield %relu : f32
    } -> tensor<1x8xf32>

    // Слой 2: y_raw = h × w2
    %y_raw = linalg.matmul
        ins(%h, %w2 : tensor<1x8xf32>, tensor<8x4xf32>)
        outs(%zero_y : tensor<1x4xf32>)
        -> tensor<1x4xf32>

    // Добавляем bias b2
    %y = linalg.add
        ins(%y_raw, %b2 : tensor<1x4xf32>, tensor<1x4xf32>)
        outs(%zero_y : tensor<1x4xf32>)
        -> tensor<1x4xf32>

    return %y : tensor<1x4xf32>
  }

} // module
