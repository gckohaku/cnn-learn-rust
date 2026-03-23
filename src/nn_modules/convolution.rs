use ndarray::{Array1, Array4, ArrayD, Axis, Ix4, Zip};

use crate::{
    cnn_transformations,
    nn_modules::{NNModule, NNNecessaryTraits},
};

#[derive(Clone, Debug)]
pub struct Convolution<T> {
    filters: Array4<T>,
    biases: Array1<T>,
    stride: usize,
    padding: usize,
    filter_size: (usize, usize),
    input_of_forward: Option<Array4<T>>,
}

impl<T> NNModule<T> for Convolution<T>
where
    T: NNNecessaryTraits,
{
    fn necessary_parameter_value(&self) -> usize {
        1
    }

    fn forward(
        &mut self,
        input: &super::NNForwardInput<'_, '_, T>,
        is_grad: bool,
    ) -> ndarray::ArrayD<T> {
        let input_4d = input.inputs[0]
            .to_owned()
            .into_dimensionality::<Ix4>()
            .unwrap();

		// (B, C_o, I_h, I_w)
        let input_shape = input_4d.shape();

		// spread_image は (C_i * F_h * F_w, B * O_h * O_w)
		// spread_filter は (C_o, C_i * F_h * F_w)
        let (spread_image, spread_filter) =
            cnn_transformations::im2col(&input_4d, &self.filters, self.stride, self.padding);

        let output_size = (
            ((input_shape[2] - self.filter_size.0 + 2 * self.padding) / self.stride) + 1,
            ((input_shape[3] - self.filter_size.1 + 2 * self.padding) / self.stride) + 1,
        );
        let bias_length = self.filters.shape()[0];

		// spread_result は (C_o, B * O_h * O_w)
        let spread_result =
            spread_filter.dot(&spread_image) + self.biases.to_shape((bias_length, 1)).unwrap();

		// (C_o, B, O_h, O_w) に形状が変化する
        let mut reshape_result = spread_result
            .to_shape([
				self.filters.shape()[0],
                input_shape[0],
                output_size.0,
                output_size.1,
            ])
            .unwrap();

		// 軸を入れ替えて (B, C_o, O_h, O_w) にする　これが出力するテンソルの形状
		reshape_result.swap_axes(0, 1);

		if is_grad {
			self.input_of_forward = Some(input_4d);
		}

        reshape_result.to_owned().into_dyn()
    }

    fn propagate_grad(
        &mut self,
        grad: Option<&ndarray::ArrayViewD<T>>,
        eta: T,
    ) -> ndarray::ArrayD<T> {
		// (B, C_o, O_h, O_w) 出力と同じ形状
        let mut grad_4d = grad
            .unwrap()
            .to_owned()
            .into_dimensionality::<Ix4>()
            .unwrap();

        let grad_shape = grad_4d.shape();
        let sample_size = grad_shape[0];
        let output_channel_value = grad_shape[1];
        let image_size = (grad_shape[2], grad_shape[3]);

		// 勾配の形状を４階テンソルに再構成する前の形状に戻す
        // 送られてきた勾配の4階テンソルは (B, C_o, O_h, O_w) なので、B と C_o を入れ替える
        grad_4d.swap_axes(0, 1);

        // (C_o, B * O_h * O_w) に平坦化
        let spread_grad = grad_4d
            .to_shape((
                output_channel_value,
                sample_size * image_size.0 * image_size.1,
            ))
            .unwrap();

		// 順伝播時の入力テンソルとフィルタも展開する
		// 順伝播時の入力: (B, C_o, I_h, I_w) -> (C_i * F_h * F_w, B * O_h * O_w)
		// フィルタ: (C_o, C_i, F_h, F_w) -> (C_o, C_i * F_h * F_w)
		let (spread_image, spread_filter) = cnn_transformations::im2col(&self.input_of_forward.to_owned().unwrap(), &self.filters, self.stride, self.padding);

		// 重みの勾配を出す (C_o, C_i * F_h * F_w) * (B * O_h * O_w, C_i * F_h * F_w)
		//                 -> (C_o, C_i * F_h * F_w)
		let spread_grad_for_filters = &spread_grad.dot(&spread_image.t());
		// 前の層に渡す勾配を出す (C_i * F_h * F_w, C_o) * (C_o, B * O_h * O_w)
		//                      -> (C_i * F_h * F_w, B * O_h * O_w)
		let spread_grad_for_before = &spread_filter.t().dot(&spread_grad);

        let filters_shape = self.filters.shape();
        // (C_o, C_i * F_h * F_w) -> (C_o, C_i, F_h, F_w)
        let reshape_grad_for_filters = spread_grad_for_filters.to_shape((filters_shape[0], filters_shape[1], filters_shape[2], filters_shape[3])).unwrap();
        Zip::from(&mut self.filters)
            .and(&reshape_grad_for_filters)
            .par_for_each(|filter, update| *filter -= eta * *update);
        Zip::from(&mut self.biases)
            .and(&spread_grad.sum_axis(Axis(0)))
            .for_each(|bias, update| *bias -= eta * *update);

        let binding = self.input_of_forward.to_owned().unwrap();
        let input_shape = &binding.shape();
        cnn_transformations::col2im(spread_grad_for_before, [self.filter_size.0, self.filter_size.1], [input_shape[0], input_shape[1], input_shape[2], input_shape[3]], self.stride, self.padding);

        ArrayD::zeros(vec![])
    }
}
