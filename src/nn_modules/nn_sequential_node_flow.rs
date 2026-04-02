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

pub struct NNSequentialNodeFlowIter<'a> {
    holder: &'a NNSequentialNodeFlow,
    index: usize,
}

impl<'a> Iterator for NNSequentialNodeFlowIter<'a> {
    type Item = &'a (usize, Vec<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.holder.sequential_process_info.len() {
            return None;
        }

        self.index += 1;
        Some(&self.holder.sequential_process_info[self.index - 1])
    }
}

impl<'a> IntoIterator for &'a NNSequentialNodeFlow {
    type Item = &'a (usize, Vec<usize>);
    type IntoIter = NNSequentialNodeFlowIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        NNSequentialNodeFlowIter {
            holder: self,
            index: 0,
        }
    }
}