use ndarray::{Array2, ArrayD, ArrayViewD, ScalarOperand};

use crate::nn_modules::{NNForwardInput, NNModule};

pub struct ReLU<T: Clone +  Send + Sync> {
    pub is_grad: bool,
    // 勾配計算のために保持するデータ
    pub(super) output_value: Option<ArrayD<T>>,
    pub(super) grad: Option<ArrayD<T>>,
}

impl<T: ScalarOperand + Send + Sync + Ord> NNModule<T> for ReLU<T> {
    fn forward<'a>(&mut self, forward_input: NNForwardInput<'_, T>) -> ArrayD<f64> {
        let input = forward_input.input;
        let mut clone_array = input.to_owned();
        clone_array.par_mapv_inplace(|x| x.max(0.0));
        self.output_value = Some(clone_array.clone());
        if self.is_grad {
            self.grad = Some(self.calc_grad(&clone_array));
        }
        return clone_array;
    }
}

impl<T: ScalarOperand + Send + Sync + PartialOrd> ReLU<T> {
    fn calc_grad(&mut self, result: &ArrayD<T>) -> ArrayD<T> {
        let grad = result.map(|y: &T| if *y > 0.0 { 1.0 } else { 0.0 });
        grad
    }
}
