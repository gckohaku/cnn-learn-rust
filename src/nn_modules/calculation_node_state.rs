#[derive(Debug)]
pub struct CalculationNodeState {
	pub id: usize,
	pub args: Vec<usize>,
}

impl CalculationNodeState {
	pub fn new(id: usize) -> Self {
		CalculationNodeState { id, args: Vec::<usize>::new() }
	}
}