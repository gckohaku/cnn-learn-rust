use ndarray::{Array1, Array2, Zip};

use crate::nn_modules::NNModule;

pub struct CrossEntropyLoss {
    pub is_grad: bool,
    // 勾配を求めるために期待値を保持しておく
    pub(super) expected_value: Option<Array2<f64>>,
}

impl NNModule for CrossEntropyLoss {
    
    type OutputArray = f64;
    type InputArray<'a> = (&'a Array2<f64>, &'a Array2<f64>);

	/// クロスエントロピー誤差の順伝播
	/// 
	/// * `input.0` - クロスエントロピー誤差を求めるときの入力値
	/// * `input.1` - クロスエントロピー誤差を求めるときのターゲット値
    fn forward<'a>(&mut self, input: (&'a Array2<f64>, &'a Array2<f64>)) -> f64 {
		let nn_result = &input.0;
		let target = &input.1;

        let ln_output = nn_result.map(|x| (x + 1e-10).ln());

        let error = -Zip::from(*target)
            .and(&ln_output)
            .fold(0.0, |t, e, o| t + e * o);

		error
    }
}
