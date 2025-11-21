use ndarray::{Array, Array2, Array4};

use crate::cnn_transformations::im2col::im2col;

mod cnn_transformations;

fn main() {
    let x: Array4<f64> = Array::from_vec((1..(3 * 3 * 4 * 4 + 1)).map(|x| x as f64).collect())
        .to_shape((3, 3, 4, 4))
        .unwrap()
        .to_owned();
    let f: Array4<f64> = Array::from_vec((-(3 * 3 * 2 * 2)..0).map(|x| x as f64).collect())
        .to_shape((3, 3, 2, 2))
        .unwrap()
        .to_owned();

    let stride = 2;
    let padding = 1;

    let (a, b) = im2col(&x, &f, stride, padding);
    println!("{:?}\n", (&a, &b));
    println!("{:?}", b.dot(&a));
}
