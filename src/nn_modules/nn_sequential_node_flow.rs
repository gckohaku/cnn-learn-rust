/// タプルの 0 番目のインデックスで表されるノードで処理されたデータが、<br>
/// 1 番目のインデックスで表されるノードへ渡されることを意味する
#[derive(Debug, Clone)]
pub struct NNSequentialNodeFlow {
    pub sequential_process_info: Vec<(usize, Vec<usize>)>,
    current_index: usize,
}

impl NNSequentialNodeFlow {
    pub fn new() -> Self {
        let sequential_process_info = Vec::<(usize, Vec<usize>)>::new();
        let current_index = 0;

        Self {
            sequential_process_info,
            current_index,
        }
    }

    pub fn add(&mut self, from: usize, to: Vec<usize>) -> usize {
        self.sequential_process_info.push((from, to));
        self.sequential_process_info.len()
    }

    pub fn add_destination_to_index(&mut self, index: usize , destination: usize) {
        self.sequential_process_info[index].1.push(destination);
    }

    pub fn next(&mut self) -> &(usize, Vec<usize>) {
        let index = self.current_index;
        self.current_index += 1;
        &self.sequential_process_info[index]
    }
}
