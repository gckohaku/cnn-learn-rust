use crate::cnn_information::TellLayerInformation;

pub struct NeuralNetworkCNN<T: TellLayerInformation> {
	pub layers_information: Vec<T>,
}