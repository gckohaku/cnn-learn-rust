use ndarray::{Array1, Array4, ArrayD};

use crate::cnn_information::{LayerInformation, LayerType};

pub struct NeuralNetworkCNN {
    pub layers_information: Vec<LayerInformation>,
    pub weights: Vec<ArrayD<f64>>,
    pub biases: Vec<Array1<f64>>,
	pub values: Vec<ArrayD<f64>>,
	pub values_after_activation: Vec<ArrayD<f64>>,
}

impl NeuralNetworkCNN {
    pub fn new(
        batch_size: usize,
        input_channel_value: usize,
        input_image_size: (usize, usize),
        layers_information: Vec<LayerInformation>,
    ) -> Self {
        let mut weights = Vec::<ArrayD<f64>>::new();
        let mut biases = Vec::<Array1<f64>>::new();
		let mut values = Vec::<ArrayD<f64>>::new();
		let mut values_after_activation = Vec::<ArrayD<f64>>::new();



        for i in 0..layers_information.len() {
			if layers_information[i].layer_type == LayerType::Convolution {
				let image_shape = values[i - 1].shape();
				// let input_image_height = 
				let outputImageHeight = () + 1;
			}
		}

        Self {
            layers_information,
            weights,
            biases,
			values,
			values_after_activation,
        }
    }

    pub fn forward(&mut self, inputs: Array4<f64>, expects: Array4<f64>) {}

    pub fn view_layers_information_state(&self) {
        dbg!(&self.layers_information);
    }
}
