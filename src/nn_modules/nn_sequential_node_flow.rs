use crate::nn_modules::nn_sequential_node_flow;

#[derive(Debug, Clone)]
pub struct NNSequentialNodeFlow {
	pub sequential_process_info: Vec<(usize, usize)>,
	current_index: usize,
}

impl NNSequentialNodeFlow {
	pub fn new() -> Self {
		let sequential_process_info = Vec::<(usize, usize)>::new();
		let current_index = 0;

		Self { sequential_process_info, current_index }
	}

	pub fn next(&self) -> &(usize, usize) {
		let index = self.current_index;
		&self.sequential_process_info[index]
	}
}