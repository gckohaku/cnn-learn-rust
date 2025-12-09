use ndarray::{Array1, Array2};

use crate::{cnn_information::{LayerInformation, OutputType}, rand::Rand};

#[derive(Debug, Clone)]
pub struct FullyConnectedNetwork {
    weights: Vec<Array2<f64>>,
    biases: Vec<Array1<f64>>,
    activations: Vec<fn(&f64) -> f64>,
    differential_activations: Vec<fn(&f64) -> f64>,
    values: Vec<Array2<f64>>,
    values_after_activation: Vec<Array2<f64>>,
    error: f64,
    deltas: Vec<Array2<f64>>,
}

impl FullyConnectedNetwork {
    pub fn new(
        batch_size: usize,
        nodes_values: Vec<usize>,
        output_type: OutputType,
    ) -> Self {
        let mut weights = Vec::<Array2<f64>>::new();
        let mut biases = Vec::<Array1<f64>>::new();
        let activations = Vec::<fn(&f64) -> f64>::new();
        let differential_activations = Vec::<fn(&f64) -> f64>::new();
        let values = Vec::<Array2<f64>>::new();
        let values_after_activation = Vec::<Array2<f64>>::new();
        let error = 0.0;
        let deltas = Vec::<Array2<f64>>::new();

        let mut r = Rand::new();

        for i in 0..nodes_values.len() {
            // 初期化段階ではいらない処理だった
            // // ノード行列のサイズは サンプル数 x 現在の層のノードの数
            // values.push(Array2::zeros([batch_size, nodes_values[i]]));
            // // 活性化関数適用後の行列のサイズは現在の層のノードのサイズと同じ
            // values_after_activation.push(values.last().unwrap().clone());

            if i > 0 {
                // 重み行列のサイズは 直前の層のノードの数 x 現在の層のノードの数
                let mut layer_weights = Array2::zeros([nodes_values[i - 1], nodes_values[i]]);
                // 重み行列を He 初期化する
                layer_weights.mapv_inplace(|_x| r.normal(0.0, (nodes_values[i - 1] as f64).sqrt()));

                weights.push(layer_weights);
                // バイアス行列のサイズは 1 x 現在の層のノードの数
                biases.push(Array1::zeros([nodes_values[i]]));
                // デルタ行列のサイズも現在の層のノードのサイズと同じ (別に初期化段階で作成する必要はなかった)
                // deltas.push(Array2::zeros([batch_size, nodes_values[i]]));
            }
        }
        Self {
            weights,
            biases,
            activations,
            differential_activations,
            values,
            values_after_activation,
            error,
            deltas,
        }
    }

	fn forward(&mut self, inputs: &Vec<Array2<f64>>, expects: &Vec<Array2<f64>>) {
		self.values.push(inputs[0].clone());
		self.values_after_activation.push(inputs[0].clone());
	}
}
