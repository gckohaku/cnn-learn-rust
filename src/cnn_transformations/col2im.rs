use std::ops::AddAssign;

use ndarray::{Array2, Array4, Dim, Slice, Zip, s};
use num_traits::{Float, FromPrimitive, Num};

pub fn col2im<T>(
    spread_value: &Array2<T>,
    filter_size: [usize; 2],
    target_shape: [usize; 4],
    stride: usize,
    padding: usize,
) -> Array4<T>
where
    T: Clone + Copy + Send + Sync + Num + Float + FromPrimitive + AddAssign,
{
    let channel_value = target_shape[1];
    let sample_value = target_shape[0];
    let input_size = (target_shape[2], target_shape[3]);

    let output_size = (
        (input_size.0 - filter_size[0] + 2 * padding) / stride + 1,
        (input_size.1 - filter_size[1] + 2 * padding) / stride + 1,
    );

    // (C_i * F_h * F_w, B * O_h * O_w) -> (C_i, F_h, F_w, B, O_h, O_w)
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

    // (C_i, F_h, F_w, B, O_h, O_w)
    // -> (B, F_h, F_w, C_i, O_h, O_w)
    // -> (B, F_h, C_i, F_w, O_h, O_w)
    // -> (B, C_i, F_h, F_w, O_h, O_w)
    cols.swap_axes(0, 3);
    cols.swap_axes(2, 3);
    cols.swap_axes(1, 2);

    // (B, C_i, I_h * 2P + S - 1, I_w * 2P + S - 1)
    let mut images = Array4::<T>::zeros((
        sample_value,
        channel_value,
        input_size.0 + 2 * padding + stride - 1,
        input_size.1 + 2 * padding + stride - 1,
    ));

    for h in 0..filter_size[0] {
        let h_limit = h + stride * output_size.0;
        for w in 0..filter_size[1] {
            let w_limit = w + stride * output_size.1;
            let mut image_slice: ndarray::ArrayBase<ndarray::ViewRepr<&mut T>, Dim<[usize; 4]>> = images.slice_mut(s![
                ..,
                ..,
                Slice::from(h..h_limit).step_by(stride.try_into().unwrap()),
                Slice::from(w..w_limit).step_by(stride.try_into().unwrap())
            ]);
            // image_slice += &cols.slice(s![.., .., h, w, .., ..]);
            Zip::from(&mut image_slice).and(&cols.slice(s![.., .., h, w, .., ..])).par_for_each(|image, col| *image += *col);
        }
    }

    let return_tensor = images.slice(s![
        ..,
        ..,
        padding..(input_size.0 + padding),
        padding..(input_size.1 + padding)
    ]);
    return_tensor.to_owned()
}
