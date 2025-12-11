use ndarray::{Array, Array1, Array2, Axis, Zip, parallel::prelude::IntoParallelRefIterator};

use crate::{
    cnn_information::{LayerInformation, OutputInformation, OutputType},
    rand::Rand,
};

use ndarray::parallel::prelude::*;

#[derive(Debug, Clone)]
pub struct FullyConnectedNetwork {
    output_information: OutputInformation,
    weights: Vec<Array2<f64>>,
    biases: Vec<Array1<f64>>,
    activations: Vec<fn(&f64) -> f64>,
    differential_activations: Vec<fn(&f64) -> f64>,
    values: Vec<Array2<f64>>,
    values_after_activation: Vec<Array2<f64>>,
    pub error: f64,
    deltas: Vec<Array2<f64>>,
}

impl FullyConnectedNetwork {
    pub fn new(
        batch_size: usize,
        nodes_values: Vec<usize>,
        output_information: OutputInformation,
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
            output_information,
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

    pub fn forward(&mut self, inputs: &Array2<f64>, expects: &Array2<f64>) {
        let batch_size = inputs.nrows();
        let output_index = self.weights.len() - 1;

        self.values.push(inputs.clone());
        self.values_after_activation.push(inputs.clone());

        for i in 0..self.weights.len() {
            let node_value = self.weights[i].ncols();

            // 線型変換およびバイアスの加算
            let transposed_value = &self.values_after_activation[i].dot(&self.weights[i])
                + &self.biases[i].to_shape((1, node_value)).unwrap();
            self.values.push(transposed_value);

            if i == output_index {
                // ここに出力層での処理
                if self.output_information.output_type == OutputType::MultiClassClassification {
                    // 多クラス分類問題では必ず softmax と交差エントロピーを用いる
                    // サンプルごとの最大値を取得
                    let max_each_sample = self.values[i + 1]
                        .map_axis(Axis(1), |row| row.fold(f64::NEG_INFINITY, |m, v| v.max(m)));

                    // サンプルごとの最大値で引いた後に、それぞれに指数関数を適用
                    let mut processed_transposed_value = Array2::zeros(self.values[i + 1].dim());

                    Zip::from(&mut processed_transposed_value)
                        .and(&self.values[i + 1])
                        .and_broadcast(&max_each_sample.to_shape((batch_size, 1)).unwrap())
                        .for_each(|result, value, max| *result = (value - max).exp());

                    // サンプルごとに指数関数の値の合計で除算する
                    let sum_exps = processed_transposed_value.sum_axis(Axis(1));
                    let mut after_softmax = Array2::zeros(processed_transposed_value.dim());

                    Zip::from(&mut after_softmax)
                        .and(&processed_transposed_value)
                        .and_broadcast(&sum_exps.to_shape((batch_size, 1)).unwrap())
                        .for_each(|result, value, sum| *result = value / sum);

                    self.values_after_activation.push(after_softmax);
                }

                break;
            }

            // 中間層では単純に ReLU 関数を利用する (max を使って実装)
            let activated_value = Array::from_shape_vec(
                (batch_size, node_value),
                self.values[i].par_iter().map(|x| x.max(0.0)).collect(),
            )
            .unwrap();

            self.values_after_activation.push(activated_value);
        }

        // 誤差を求める ここでは、サンプル数で除算しない
        if self.output_information.output_type == OutputType::MultiClassClassification {
            let ln_output =
                self.values_after_activation[output_index + 1].map(|x| (x + 1e-10).ln());

            self.error = -Zip::from(expects)
                .and(&ln_output)
                .fold(self.error, |t, e, o| t + e * o);

            // dbg!(&self.values_after_activation[output_index + 1]);
            // println!();
            // dbg!(&expects);
            // println!();
            // dbg!(self.error);
        }
    }
}
