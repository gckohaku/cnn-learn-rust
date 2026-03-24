use std::fmt::Debug;

use ndarray::{ArrayD, Zip};

use crate::nn_modules::{NNForwardInput, NNModule, NNNecessaryTraits};

#[derive(Debug, Clone)]
pub struct ReLU<T> {
    // 勾配計算のために保持するデータ
    pub(super) output_value: Option<ArrayD<T>>,
    pub(super) grad: Option<ArrayD<T>>,
}

impl<T> NNModule<T> for ReLU<T>
where
    T: NNNecessaryTraits,
{
    fn necessary_parameter_value(&self) -> usize {
        1
    }

    fn forward<'a>(&mut self, forward_input: &NNForwardInput<'_, '_, T>, is_grad: bool) -> ArrayD<T> {
        let input = &forward_input.inputs[0];
        let mut clone_array = input.to_owned();
        clone_array.par_mapv_inplace(|x| if x > T::ZERO { x } else { T::ZERO });
        self.output_value = Some(clone_array.clone());
        if is_grad {
            // self.grad = Some(self.calc_grad(&clone_array));
            self.grad = Some(clone_array.clone());
        }
        
        clone_array
    }

    fn propagate_grad(&mut self, grad: Option<&ndarray::ArrayViewD<T>>, _eta: T) -> ArrayD<T> {
        let mut before_grad = grad.unwrap().to_owned();
        Zip::from(&mut before_grad).and(&self.grad.clone().unwrap()).par_for_each(|b: &mut T, s: &T| *b = if *s > T::ZERO {*b} else {T::ZERO});
        before_grad
    }
}

// impl<T> ReLU<T>
// where
//     T: NNNecessaryTraits,
// {
//     fn calc_grad(&mut self, result: &ArrayD<T>) -> ArrayD<T> {
//         let mut grad = result.clone();
//         grad.par_mapv_inplace(|y: T| if y > T::ZERO { T::ONE } else { T::ZERO });
//         grad
//     }
// }
