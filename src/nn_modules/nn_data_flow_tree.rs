use ndarray::ArrayViewD;

use crate::nn_modules::{NNDataFlowNodeIndexInfo, NNModule, nn_sequential_node_flow::NNSequentialNodeFlow};
use std::fmt::Debug;

pub trait NNTreeNode {
    const NECESSARY_VARIABLE_VALUE: usize;
}

#[derive(Debug, Clone)]
pub struct NNDataFlowTree<'a, T> {
    pub modules: Vec<Box<&'a dyn NNModule<T>>>,
    adjacency_list: Vec<Vec<usize>>,
    current_count: usize,
    root_node_indices: Vec<usize>,
    // 計算結果を使用する回数　この値が 0 になるまでは clone する
    use_calculation_result_times: Vec<usize>,
    variables_stocks: Vec<Vec<ArrayViewD<'a, T>>>,
    input_variables: Vec<Vec<ArrayViewD<'a, T>>>,
    sequential_flow: NNSequentialNodeFlow,
}

// impl<'a, T> NNModule<T> for NNDataFlowTree<T>
// where
//     T: Clone + Zero,
// {
//     fn forward(&mut self, input: &NNForwardInput<'_, '_, T>) -> ArrayD<T> {

//     }
// }

impl<'a, T> NNDataFlowTree<'a, T> {
    pub fn new() -> Self {
        let modules = Vec::<Box<&'_ dyn NNModule<T>>>::new();
        let adjacency_list = Vec::<Vec<usize>>::new();
        let current_count = 0;
        let root_node_indices = Vec::<usize>::new();
        let use_calculation_result_times = Vec::<usize>::new();
        let variables_stocks = Vec::<Vec<ArrayViewD<'a, T>>>::new();
        let input_variables = Vec::<Vec<ArrayViewD<'a, T>>>::new();
        let sequential_flow = NNSequentialNodeFlow::new();

        Self {
            modules,
            adjacency_list,
            current_count,
            root_node_indices,
            use_calculation_result_times,
            variables_stocks,
            input_variables,
            sequential_flow,
        }
    }

    pub fn add_from_root(&mut self, module: &'a dyn NNModule<T>) -> NNDataFlowNodeIndexInfo {
        self.modules.push(Box::new(module));
        let index = self.current_count;
        self.current_count += 1;

        self.adjacency_list.push(Vec::<usize>::new());
        self.root_node_indices.push(index);

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

    fn calc_sequential_node_flow(&mut self) {
        // 隣接リストは既に生成済みであるとする
        // ルートノードに繋がっているモジュールには既に適切な変数が指定されているものとする
        for index in &self.root_node_indices {
            // そういやルートノードのインデックスの表現はどうしようか？
        }
    }
}
