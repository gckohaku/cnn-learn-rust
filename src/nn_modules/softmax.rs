use ndarray::{Array2, Axis, Zip};

use crate::nn_modules::NNModule;

pub struct Softmax {
	// 勾配を求めるとき用に出力値を保持しておく
	pub(super) output_value: Option<Array2<f64>>,
}

impl NNModule for Softmax {
    type InputArray = Array2<f64>;
    type OutputArray = Array2<f64>;

    fn forward(&mut self, input: Array2<f64>) -> Array2<f64> {
		let batch_size = input.nrows();

        let max_each_sample = input
            .map_axis(Axis(1), |row| row.fold(f64::NEG_INFINITY, |m, v| v.max(m)));

        // サンプルごとの最大値で引いた後に、それぞれに指数関数を適用
        let mut processed_transposed_value = Array2::zeros(input.dim());

        Zip::from(&mut processed_transposed_value)
            .and(&input)
            .and_broadcast(&max_each_sample.to_shape((batch_size, 1)).unwrap())
            .for_each(|result, value, max| *result = (value - max).exp());

        // サンプルごとに指数関数の値の合計で除算する
        let sum_exps = processed_transposed_value.sum_axis(Axis(1));
        let mut after_softmax = Array2::zeros(processed_transposed_value.dim());

        Zip::from(&mut after_softmax)
            .and(&processed_transposed_value)
            .and_broadcast(&sum_exps.to_shape((batch_size, 1)).unwrap())
            .for_each(|result, value, sum| *result = value / sum);

        #[cfg(debug_assertions)]
        {
            dbg!(&processed_transposed_value, &after_softmax);
        }

		self.output_value = Some(after_softmax.clone());

        after_softmax
    }
}
