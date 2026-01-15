use ndarray::{Array2, ArrayD, ArrayViewD};

use crate::nn_modules::NNModule;

pub struct ReLU {
    pub is_grad: bool,
    // 勾配計算のために保持するデータ
    pub(super) output_value: Option<ArrayD<f64>>,
    pub(super) grad: Option<ArrayD<f64>>,
}

impl NNModule for ReLU {
    type OutputArray = ArrayD<f64>;

    fn forward<'a>(&mut self, input: ArrayViewD<f64>) -> ArrayD<f64> {
        let mut clone_array = input.to_owned();
        clone_array.par_mapv_inplace(|x| x.max(0.0));
        self.output_value = Some(clone_array.clone());
        if self.is_grad {
            self.grad = Some(self.calc_grad(&clone_array));
        }
        return clone_array;
    }
}

impl ReLU {
    fn calc_grad(&mut self, result: &ArrayD<f64>) -> ArrayD<f64> {
        let grad = result.map(|y: &f64| if *y > 0.0 { 1.0 } else { 0.0 });
        grad
    }
}
