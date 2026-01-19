use ndarray::{Array2, ArrayD, ArrayViewD, ScalarOperand};

use crate::nn_modules::{HasOne, HasZero, NNForwardInput, NNModule};

pub struct ReLU<T: Clone + Send + Sync> {
    pub is_grad: bool,
    // 勾配計算のために保持するデータ
    pub(super) output_value: Option<ArrayD<T>>,
    pub(super) grad: Option<ArrayD<T>>,
}

impl<T> NNModule<T> for ReLU<T>
where
    T: ScalarOperand + Send + Sync + Ord + HasZero<Output = T> + HasOne<Output = T>,
{
    fn forward<'a>(&mut self, forward_input: NNForwardInput<'_, T>) -> ArrayD<T> {
        let input = forward_input.input;
        let mut clone_array = input.to_owned();
        clone_array.par_mapv_inplace(|x| x.max(T::get_zero()));
        self.output_value = Some(clone_array.clone());
        if self.is_grad {
            self.grad = Some(self.calc_grad(&clone_array));
        }
        return clone_array;
    }
}

impl<T> ReLU<T>
where
    T: ScalarOperand + Send + Sync + Ord + HasZero<Output = T> + HasOne<Output = T>,
{
    fn calc_grad(&mut self, result: &ArrayD<T>) -> ArrayD<T> {
        let grad = result.map(|y: &T| {
            if *y > T::get_zero() {
                T::get_one()
            } else {
                T::get_zero()
            }
        });
        grad
    }
}
