mod cnn_activations;
mod cnn_information;
mod cnn_network;
mod cnn_transformations;
mod convolution_network;
mod fully_connected_network;
mod mnist_test;
mod rand;
mod simple_dataset;
mod test_cnn_network;
mod type_utilities;
mod utilities;
mod inspection_cnn;
mod nn_modules;

fn main() {
    mnist_test::mnist_process();
    // inspection_cnn::inspection();
}
