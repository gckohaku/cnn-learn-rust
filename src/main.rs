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
mod test_nn_modules;
mod mnist_test_cnn;

fn main() {
    mnist_test_cnn::mnist_process();
    // inspection_cnn::inspection();

    // test_nn_modules::run_test();
}
