use std::time::{SystemTime, UNIX_EPOCH};

use ndarray::{Array1, Array2};

use crate::{cnn_information::{LayerInformation, OutputType}, rand::Rand};
use rand_pcg::rand_core::SeedableRng;

#[derive(Debug, Clone)]
pub struct FullyConnectedNetwork {
    layers_information: Vec<LayerInformation>,
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
    pub fn new(batch_size: usize, nodes_values: Vec<usize>, layers_information: Vec<LayerInformation>) -> Self {
        let mut weights = Vec::<Array2<f64>>::new();
        let mut biases = Vec::<Array1<f64>>::new();
        let mut activations = Vec::<fn(&f64) -> f64>::new();
        let mut differential_activations = Vec::<fn(&f64) -> f64>::new();
		let mut values = Vec::<Array2<f64>>::new();
		let mut values_after_activation = Vec::<Array2<f64>>::new();
		let error = 0.0;
		let mut deltas = Vec::<Array2<f64>>::new();
		
		let mut r = Rand::new();

		for i in 0..nodes_values.len() {
			// ノード行列のサイズは サンプル数 x 現在の層のノードの数
			values.push(Array2::zeros([batch_size, nodes_values[i]]));
			// 活性化関数適用後の行列のサイズは現在の層のノードのサイズと同じ
			values_after_activation.push(values.last().unwrap().clone());

			if i > 0 {
				// 重み行列のサイズは 直前の層のノードの数 x 現在の層のノードの数
				let mut layer_waights = Array2::zeros([nodes_values[i - 1], nodes_values[i]]);
				// 重み行列を He 初期化する
				layer_waights.map_inplace(|x| *x = r.normal(0.0, (nodes_values[i - 1] as f64).sqrt()));
			}
		}
        Self {
            layers_information,
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
}
