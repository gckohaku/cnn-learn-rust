use ndarray::{Array2, Array4, Slice, s};

pub fn col2im(
    spread_value: &Array2<f64>,
    filter_size: [usize; 2],
    target_shape: [usize; 4],
    stride: usize,
    padding: usize,
) -> Array4<f64> {
    let channel_value = target_shape[1];
    let sample_value = target_shape[0];
    let input_size = (target_shape[0], target_shape[1]);

    let output_size = (
        (input_size.0 - filter_size[0] + 2 * padding) / stride + 1,
        (input_size.1 - filter_size[1] + 2 * padding) / stride + 1,
    );

    let mut cols = spread_value
        .to_shape((
            channel_value,
            filter_size[0],
            filter_size[1],
            sample_value,
            output_size.0,
            output_size.1,
        ))
        .unwrap();
    cols.swap_axes(0, 3);
    cols.swap_axes(2, 3);
    cols.swap_axes(1, 2);

    let mut images = Array4::<f64>::zeros((
        sample_value,
        channel_value,
        input_size.0 + 2 * padding + stride - 1,
        input_size.1 + 2 * padding + stride - 1,
    ));

    for h in 0..filter_size[0] {
        let h_limit = h + stride * output_size.0;
        for w in 0..filter_size[1] {
            let w_limit = w + stride * output_size.1;
            let mut image_slice = images.slice_mut(s![
                ..,
                ..,
                Slice::from(h..h_limit).step_by(stride.try_into().unwrap()),
                Slice::from(w..w_limit).step_by(stride.try_into().unwrap())
            ]);
			image_slice += &cols.slice(s![.., .., h, w, .., ..]);
        }
    }

	let return_tensor = images.slice(s![.., .., padding..(input_size.0 + padding), padding..(input_size.1 + padding)]);
	return_tensor.to_owned()
}
