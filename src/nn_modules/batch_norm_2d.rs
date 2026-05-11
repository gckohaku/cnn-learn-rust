use ndarray::{Array1, Array2, ArrayD, ArrayViewD, Axis, Ix4, Zip, parallel::prelude::{IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator}};

use crate::{
    impl_as_any_with_mut,
    nn_modules::{NNForwardInput, NNModule, NNNecessaryTraits},
};

#[derive(Debug, Clone)]
pub struct BatchNorm2d<T> {
    pub(crate) beta_average: Array1<T>,
    pub(crate) gamma_variance: Array1<T>,
    // 学習に必要なもの
    pub(crate) std_input: Array2<T>,
    pub(crate) input_minus_mu: Array2<T>,
    pub(crate) input_variance: Array1<T>,

    // 以下、推論時に使用するもの
    pub(crate) running_average: Array1<T>,
    pub(crate) running_variance: Array1<T>,
}

impl<T> NNModule<T> for BatchNorm2d<T>
where
    T: NNNecessaryTraits,
{
    fn necessary_parameter_value(&self) -> usize {
        1
    }

    fn forward(
        &mut self,
        input: &NNForwardInput<'_, '_, T>,
        is_grad: bool,
    ) -> Result<ArrayD<T>, &'static str> {
        let input_images = &input.inputs[0];
        let mut input_images_4d = input_images.clone().into_dimensionality::<Ix4>().unwrap();
        input_images_4d.permute_axes([0, 2, 3, 1]);
        let input_permute_shape = input_images_4d.shape();
        let matrix_row_value =
            input_permute_shape[0] * input_permute_shape[1] * input_permute_shape[2];
        let input_images_2d = input_images_4d
            .to_shape([matrix_row_value, input_permute_shape[3]])
            .unwrap();

        let mean_per_channel = if is_grad {
            input_images_2d.mean_axis(Axis(0)).unwrap()
        } else {
            self.running_average.clone()
        };
        let variance_per_channel = if is_grad {
            input_images_2d.var_axis(Axis(0), T::ZERO)
        } else {
            self.running_variance.clone()
        };

        if is_grad {
            self.running_average = &self.running_average * T::from(0.999).unwrap()
                + &mean_per_channel * T::from(0.001).unwrap();
            self.running_variance = &self.running_variance * T::from(0.999).unwrap()
                + &variance_per_channel * T::from(0.001).unwrap();
            self.input_minus_mu = &input_images_2d - &mean_per_channel;
            self.input_variance = &variance_per_channel + T::from(1e-5).unwrap();
        }

        let inverse_std_per_channel = Zip::from(&variance_per_channel)
            .par_map_collect(|var| T::ONE / T::sqrt(*var + T::from(1e-5).unwrap()));

        let norm_images_2d = inverse_std_per_channel * (input_images_2d - mean_per_channel);

        if is_grad {
            self.std_input = norm_images_2d.clone();
        }

        let result_2d = (norm_images_2d * &self.gamma_variance) + &self.beta_average;

        let mut result_4d = result_2d
            .to_shape([
                input_permute_shape[0],
                input_permute_shape[1],
                input_permute_shape[2],
                input_permute_shape[3],
            ])
            .unwrap();
        result_4d.permute_axes([0, 3, 1, 2]);

        Ok(result_4d.to_owned().into_dyn())
    }

    fn propagate_grad(&mut self, grad: Option<&ArrayViewD<T>>, eta: T) -> ArrayD<T> {
        let mut grad_4d = grad.unwrap().clone().into_dimensionality::<Ix4>().unwrap();
        grad_4d.permute_axes([0, 2, 3, 1]);
        let grad_permute_shape = grad_4d.shape();

        let matrix_row_value =
            grad_permute_shape[0] * grad_permute_shape[1] * grad_permute_shape[2];
        let grad_2d = grad_4d
            .to_shape([matrix_row_value, grad_permute_shape[3]])
            .unwrap();

        let grad_for_beta = grad_2d.sum_axis(Axis(0));
        let grad_for_gamma = (&grad_2d * &self.std_input).sum_axis(Axis(0));

        _ = &self.input_variance.par_iter_mut().for_each(|x| *x = x.sqrt());
        let grad_for_input_2d = &self.gamma_variance / (&self.input_variance * T::from(matrix_row_value).unwrap()) * (&grad_2d * T::from(matrix_row_value).unwrap() - &grad_for_beta - &grad_for_gamma * &self.std_input);
        // let mut grad_for_input_2d = Array2::<T>::zeros((matrix_row_value, grad_permute_shape[3]));
        // dbg!("input");
        // Zip::from(grad_for_input_2d.axis_iter_mut(Axis(1))).and(&)
        //     .and(&self.input_variance)
        //     .and(self.input_minus_mu.axis_iter(Axis(1)))
        //     .par_for_each(
        //         |mut grad_input, grad_std, grad_var, grad_mu, input_var, minus| {
        //             let a = &grad_std * (T::ONE / *input_var);
        //             let b = &minus
        //                 * (*grad_var
        //                     * (T::from(2.0).unwrap() / T::from(matrix_row_value).unwrap()));
        //             let c = Array1::<T>::from_elem(
        //                 matrix_row_value,
        //                 T::ONE / T::from(matrix_row_value).unwrap(),
        //             ) * (*grad_mu);
        //             grad_input.assign(&(a + b + c));
        //         },
        //     );

        self.gamma_variance -= &(grad_for_gamma * eta);
        self.beta_average -= &(grad_for_beta * eta);

        let mut grad_for_input_4d = grad_for_input_2d
            .to_shape([
                grad_permute_shape[0],
                grad_permute_shape[1],
                grad_permute_shape[2],
                grad_permute_shape[3],
            ])
            .unwrap();

        grad_for_input_4d.permute_axes([0, 3, 1, 2]);

        grad_for_input_4d.to_owned().into_dyn()
    }

    impl_as_any_with_mut!();
}
