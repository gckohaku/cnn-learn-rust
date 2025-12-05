use ndarray::Array3;

use crate::rand::Rand;

mod cnn_activations;
mod cnn_information;
mod cnn_network;
mod cnn_transformations;
mod convolution_network;
mod type_utilities;
mod fully_connected_network;
mod rand;

fn main() {
	let mut arr = Array3::<f64>::zeros([2, 3 ,4]);

	let mut r = Rand::new();

	arr.mapv_inplace(|_x| r.rand_f64());

	println!("{:?}", arr);
}
