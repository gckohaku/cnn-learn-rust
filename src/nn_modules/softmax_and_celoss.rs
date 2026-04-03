use std::{fmt::Debug, iter::Sum};

use ndarray::{Array2, ArrayD, Ix2, Zip, parallel::prelude};

use crate::{
    impl_as_any_with_mut,
    nn_modules::{CrossEntropyLoss, NNForwardInput, NNModule, NNNecessaryTraits, Softmax},
};

#[derive(Debug, Clone)]
pub struct SoftmaxAndCELoss<T> {
    pub softmax: Softmax<T>,
    pub cross_entropy_loss: CrossEntropyLoss<T>,
    pub is_test: bool,
    // テストの結果を保持 (correct_value, test_value)
    pub test_correct_value: usize,
    // 勾配を求める時に利用
    pub(super) grad: Option<Array2<T>>,
}

impl<T> NNModule<T> for SoftmaxAndCELoss<T>
where
    T: NNNecessaryTraits,
{
    fn necessary_parameter_value(&self) -> usize {
        1
    }

    fn forward<'a, 'b>(&mut self, input: &NNForwardInput<'a, 'b, T>, is_grad: bool) -> ArrayD<T> {
        let input_values = input;

        let softmax_result = self.softmax.forward(&input_values, is_grad);

        let softmax_result_view = softmax_result.view();

        let target = &input_values.target;

        let ce_loss_input = &NNForwardInput {
            inputs: vec![softmax_result_view],
            target: Some(target.clone().unwrap()),
        };
        let loss = self.cross_entropy_loss.forward(ce_loss_input, is_grad);

        let softmax_result_2d = softmax_result.into_dimensionality::<Ix2>().unwrap();
        let target_2d = input
            .target
            .to_owned()
            .unwrap()
            .into_dimensionality::<Ix2>()
            .unwrap();

        if is_grad {
            // self.grad = Some(&softmax_result.to_owned().into_dimensionality::<Ix2>().unwrap() - &target.to_owned().unwrap().into_dimensionality::<Ix2>().unwrap());
            self.grad = Some(&softmax_result_2d - &target_2d);
        }

        if self.is_test {
            self.test_correct_value += Zip::from(softmax_result_2d.rows())
                .and(target_2d.rows())
                .par_fold(
                    || 0usize,
                    |sum, s, t| {
                        if s.len() != 10 {
                            panic!("in validation test: vector size is not 10\nactually vector size: {}", s.len());
                        }
                        let most_index = s
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                            .map(|(index, _)| index)
                            .unwrap();
                        if t[most_index] > T::ZERO {
                            sum + 1usize
                        } else {
                            sum + 0usize
                        }
                    },
                    |sum, res| {sum + res},
                );
        }

        loss
    }

    fn propagate_grad(&mut self, grad: Option<&ndarray::ArrayViewD<T>>, _eta: T) -> ArrayD<T> {
        match grad {
            None => self.grad.to_owned().unwrap().into_dyn(),
            Some(g) => {
                let self_2d = self
                    .grad
                    .to_owned()
                    .unwrap()
                    .into_dimensionality::<Ix2>()
                    .unwrap();
                let propagated_2d = &g.to_owned().into_dimensionality::<Ix2>().unwrap();
                (self_2d * propagated_2d).into_dyn()
            }
        }
    }

    impl_as_any_with_mut!();
}
