use crate::{cnn_information::{InformationList, LayerInformation}, type_utilities::{Here, Member}};

pub struct NeuralNetworkCNN<T> where T:  {
	pub layers_information: Vec<T>,
}
