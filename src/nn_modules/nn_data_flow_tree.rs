use ndarray::{ArrayD, ArrayViewD};

use crate::nn_modules::{
    NNDataFlowNodeIndexInfo, NNModule, calculation_node_state::CalculationNodeState, input_tensor::InputTensor, nn_sequential_node_flow::NNSequentialNodeFlow
};
use std::{collections::VecDeque, fmt::Debug};

pub trait NNNodeNeedsParameter {
    fn parameter_value(&self) -> usize;
}

#[derive(Debug, Clone)]
pub struct NNDataFlowTree<'a, T> {
    pub modules: Vec<Box<&'a dyn NNModule<T>>>,
    adjacency_list: Vec<Vec<usize>>,
    current_count: usize,
    root_node_parameters: Vec<usize>,
    // 計算結果を使用する回数　この値が 0 になるまでは clone する
    use_calculation_result_times: Vec<usize>,
    variables_stocks: Vec<Vec<ArrayViewD<'a, T>>>,
    input_variables_indices: Vec<Vec<usize>>,
    sequential_flow: NNSequentialNodeFlow,
}

// impl<'a, T> NNModule<T> for NNDataFlowTree<T>
// where
//     T: Clone + Zero,
// {
//     fn forward(&mut self, input: &NNForwardInput<'_, '_, T>) -> ArrayD<T> {

//     }
// }

impl<'a, T> NNDataFlowTree<'a, T>
where
    T: Clone + std::fmt::Debug,
{
    pub fn new() -> Self {
        let modules = Vec::<Box<&'_ dyn NNModule<T>>>::new();
        let adjacency_list = Vec::<Vec<usize>>::new();
        let current_count = 0;
        let root_node_indices = Vec::<usize>::new();
        let use_calculation_result_times = Vec::<usize>::new();
        let variables_stocks = Vec::<Vec<ArrayViewD<'a, T>>>::new();
        let input_variables_indices = Vec::<Vec<usize>>::new();
        let sequential_flow = NNSequentialNodeFlow::new();

        Self {
            modules,
            adjacency_list,
            current_count,
            root_node_parameters: root_node_indices,
            use_calculation_result_times,
            variables_stocks,
            input_variables_indices,
            sequential_flow,
        }
    }

    pub fn add_from_root(
        &mut self,
        module: &'a (impl NNModule<T> + NNNodeNeedsParameter),
    ) -> NNDataFlowNodeIndexInfo {
        let parameter_value = module.parameter_value();
        for i in 0..parameter_value {
            self.modules.push(Box::new(&InputTensor {}));
            self.adjacency_list.push(Vec::<usize>::new());
            self.adjacency_list[self.current_count + i].push(self.current_count + parameter_value);
        }
        self.current_count += 1;
        
        self.modules.push(Box::new(module));
        let index = self.current_count;
        self.current_count += 1;

        self.adjacency_list.push(Vec::<usize>::new());
        
        for i in 1..=parameter_value {
            self.root_node_parameters.push(index - i);
        }

        NNDataFlowNodeIndexInfo { index }
    }

    pub fn add(
        &mut self,
        from: &NNDataFlowNodeIndexInfo,
        module: &'a dyn NNModule<T>,
    ) -> NNDataFlowNodeIndexInfo {
        self.modules.push(Box::new(module));
        let from_index = from.index;
        let to_index = self.current_count;
        self.current_count += 1;

        self.adjacency_list.push(Vec::<usize>::new());
        self.adjacency_list[from_index].push(to_index);

        NNDataFlowNodeIndexInfo { index: to_index }
    }

    pub fn calc_sequential_node_flow(&mut self) {
        // 隣接リストは既に生成済みであるとする
        // ルートノードに繋がっているモジュールには既に適切な変数が指定されているものとする

        let mut calculation_queue = VecDeque::<CalculationNodeState>::new();

        self.input_variables_indices.resize(self.modules.len(), Vec::<usize>::new());

        // 入力パラメータがどのノードに渡されるのかを表現する
        for index in &self.root_node_parameters {
            for destination in &self.adjacency_list[*index] {
                self.sequential_flow.add(*index, *destination);
                self.input_variables_indices[*destination].push(*index);

                match calculation_queue.iter().position(|state| state.id == *destination) {
                    None => {calculation_queue.push_back(CalculationNodeState { id: *destination, args: vec![*index] })},
                    Some(pos) => calculation_queue.get_mut(pos).unwrap().args.push(*index),
                }
            }
        }

        // キューが無くなるまで計算順序割り出し処理を継続する
        while calculation_queue.is_empty() == false {
            let current_state = calculation_queue.pop_front().unwrap();
            let current_id = current_state.id;
            let current_module = &self.modules[current_id];
            // let necessary_parameter_value = ;
            // parameter_value メソッドは NNModule トレイトで要請した方がいいかもしれない
            // パラメータを必要としない場合は、結果を 0 とすればよい
        }
    }
}
