use std::fmt::Debug;

use ndarray::{Array1, Array2, ArrayD, ArrayViewD, Axis, Ix2, Zip};

use crate::nn_modules::{NNForwardInput, NNModule, NNNecessaryTraits};

/// アフィン変換を行うニューラルネットワークモジュール
#[derive(Clone)]
pub struct Linear<T> {
    pub weights: Array2<T>,
    pub biases: Array1<T>,
    // 重みを更新するために保持するデータ
    // いうて grad に入れているからいらないかもしれない
    pub(super) input_value: Option<Array2<T>>,
    pub(super) grad: Option<Array2<T>>,
}

impl<T> Linear<T> {
    #[cfg(debug_assertions)]
    pub fn debug_set_weights(&mut self, w: Array2<T>) {
        self.weights = w;
    }

    #[cfg(debug_assertions)]
    pub fn debug_set_biases(&mut self, b: Array1<T>) {
        self.biases = b;
    }
}

impl<T> NNModule<T> for Linear<T>
where
    T: NNNecessaryTraits,
{
    fn necessary_parameter_value(&self) -> usize {
        1
    }

    fn forward(&mut self, forward_input: &NNForwardInput<'_, '_, T>, is_grad: bool) -> ArrayD<T> {
        let input = &forward_input.inputs[0];
        let input_2d = input.clone().into_dimensionality::<Ix2>().unwrap();

        // Linear 層では、勾配計算のために入力行列の転置を保持しておけばよい
        if is_grad == true {
            self.grad = Some(input_2d.t().to_owned());
        }

        let result = input_2d.dot(&self.weights) + &self.biases;
        result.into_dyn()
    }

    fn propagate_grad(&mut self, grad: Option<&ArrayViewD<T>>, eta: T) -> ArrayD<T> {
        let before_grad = &grad
            .unwrap()
            .to_owned()
            .into_dimensionality::<Ix2>()
            .unwrap();

        // 伝播させる勾配の計算
        let propagated_to_before = before_grad.dot(&self.weights.t());

        // 重み、バイアスの更新
        Zip::from(&mut self.weights)
            .and(&self.grad.to_owned().unwrap().dot(&before_grad.to_owned()))
            .par_for_each(|weight, update| *weight -= eta * *update);
        Zip::from(&mut self.biases)
            .and(&before_grad.sum_axis(Axis(0)))
            .for_each(|bias, update| *bias -= eta * *update);

        propagated_to_before.into_dyn()
    }
}

impl<T> Debug for Linear<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Linear {{")?;
        writeln!(f, "    weights shape: {:?}", self.weights.shape())?;
        writeln!(f, "    biases shape: {:?}", self.biases.shape())?;
        writeln!(f, "    grad: {:?}", self.grad)?;
        writeln!(f, "}}")?;
        Ok(())
    }
}
