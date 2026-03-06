use ndarray::{ArrayD};
use num_traits::{ConstOne, ConstZero, Float};

use crate::nn_modules::{
    NNDataFlowNodeIndexInfo, NNForwardInput, NNModule, NNModuleType, NNNecessaryTraits,
    calculation_node_state::CalculationNodeState, input_tensor::InputTensor,
    nn_sequential_node_flow::NNSequentialNodeFlow,
};
use std::{collections::VecDeque, fmt::Debug, marker::PhantomData, usize};

#[derive(Debug, Clone)]
pub struct NNDataFlowTree<T>
where
    T: Clone,
{
    pub modules: Vec<NNModuleType<T>>,
    adjacency_list: Vec<Vec<usize>>,
    inverse_adjacency_list: Vec<Vec<usize>>,
    current_count: usize,
    root_node_parameters: Vec<usize>,
    // 計算結果を使用する回数　この値が 0 になるまでは clone する
    // use_calculation_result_times: Vec<usize>,
    variables_stocks: Vec<Vec<ArrayD<T>>>,
    // input_variables_indices: Vec<Vec<usize>>,
    sequential_flow: NNSequentialNodeFlow,
}

impl<'a, T> NNModule<T> for NNDataFlowTree<T>
where
    T: NNNecessaryTraits,
{
    fn necessary_parameter_value(&self) -> usize {
        usize::MAX
    }

    fn forward(&mut self, input: &NNForwardInput<'_, '_, T>) -> ArrayD<T> {
        let target = input.target.as_ref().unwrap();
        let mut loop_result = ArrayD::<T>::zeros(vec![]);

        // まず、すべての入力引数を適切なところに保管する
        self.variables_stocks.resize(self.modules.len(), vec![]);
        for i in 0..self.root_node_parameters.len() {
            let set_index = self.root_node_parameters[i];
            self.variables_stocks[set_index].push(input.inputs[i].to_owned());
        }

        // sequential_flow をそのままループすれば大丈夫なはず
        for flow in self.sequential_flow.into_iter() {
            let index = flow.0;
            let destinations = &flow.1;
            let module = &mut self.modules[index];

            if module.necessary_parameter_value() !=  self.variables_stocks[index].len() {
                panic!("Mismatch parameter value:\n    necessary: {},\n    actual: {}.", module.necessary_parameter_value(), self.variables_stocks[index].len());
            }

            loop_result = module.forward(&NNForwardInput { inputs: self.variables_stocks[index].iter().map(|m| m.view()).collect(), target: Some(target.clone()) });

            for dst in destinations {
                self.variables_stocks[*dst].push(loop_result.clone());
            }
        }

        loop_result
    }
}

impl<T> NNDataFlowTree<T>
where
    T: Clone + std::fmt::Debug,
{
    pub fn new() -> Self {
        let modules = Vec::<NNModuleType<T>>::new();
        let adjacency_list = Vec::<Vec<usize>>::new();
        let inverse_adjacency_list = Vec::<Vec<usize>>::new();
        let current_count = 0;
        let root_node_indices = Vec::<usize>::new();
        // let use_calculation_result_times = Vec::<usize>::new();
        let variables_stocks = Vec::<Vec<ArrayD<T>>>::new();
        // let input_variables_indices = Vec::<Vec<usize>>::new();
        let sequential_flow = NNSequentialNodeFlow::new();

        Self {
            modules,
            adjacency_list,
            inverse_adjacency_list,
            current_count,
            root_node_parameters: root_node_indices,
            // use_calculation_result_times,
            variables_stocks,
            // input_variables_indices,
            sequential_flow,
        }
    }

    pub fn add_from_root(&mut self, module: NNModuleType<T>) -> NNDataFlowNodeIndexInfo
    where
        T: Send + Sync + Debug + Float + ConstOne + ConstZero + 'static,
    {
        let parameter_value = module.necessary_parameter_value();
        for i in 0..parameter_value {
            self.modules
                .push(NNModuleType::InputTensor(InputTensor::<T> {
                    phantom: PhantomData,
                }));
            self.adjacency_list.push(Vec::<usize>::new());
            self.adjacency_list[self.current_count + i].push(self.current_count + parameter_value);
        }
        self.current_count += 1;

        self.modules.push(module);
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
        module: NNModuleType<T>,
    ) -> NNDataFlowNodeIndexInfo {
        self.modules.push(module);
        let from_index = from.index;
        let to_index = self.current_count;
        self.current_count += 1;

        self.adjacency_list.push(Vec::<usize>::new());
        self.adjacency_list[from_index].push(to_index);

        NNDataFlowNodeIndexInfo { index: to_index }
    }

    pub fn calc_sequential_node_flow(&mut self)
    where
        NNModuleType<T>: NNModule<T>,
    {
        let mut calculation_queue = VecDeque::<CalculationNodeState>::new();

        let queue_match =
            |source: usize, destination: usize, queue: &mut VecDeque<CalculationNodeState>| {
                match queue.iter().position(|state| state.id == destination) {
                    None => queue.push_back(CalculationNodeState {
                        id: destination,
                        args: vec![source],
                    }),
                    Some(pos) => queue.get_mut(pos).unwrap().args.push(source),
                }
            };

        // 入力パラメータがどのノードに渡されるのかを表現する
        for index in &self.root_node_parameters {
            for destination in &self.adjacency_list[*index] {
                self.sequential_flow.add(*index, vec![*destination]);
                // self.input_variables_indices[*destination].push(*index);

                queue_match(*index, *destination, &mut calculation_queue);
            }
        }

        // キューが無くなるまで計算順序割り出し処理を継続する
        while calculation_queue.is_empty() == false {
            let current_state = calculation_queue.pop_front().unwrap();
            let current_id = current_state.id;
            let current_module = &self.modules[current_id];
            let necessary_parameter_value = current_module.necessary_parameter_value();
            let current_parameter_value = current_state.args.len();

            if current_parameter_value > necessary_parameter_value {
                panic!(
                    "Parameter value is Exceeded\n    necessary: {}\n    current: {}",
                    necessary_parameter_value, current_parameter_value
                );
            }
            if current_parameter_value < necessary_parameter_value {
                calculation_queue.push_back(current_state);
                continue;
            }

            // パラメータの数がちょうど求められていた数の場合は、self.sequential_flow に情報を入れ、calculation_queue に必要な情報を入れる
            let destinations = &self.adjacency_list[current_id];
            let push_index = self.sequential_flow.add(current_id, vec![]) - 1;

            for dst in destinations {
                queue_match(current_id, *dst, &mut calculation_queue);
                self.sequential_flow
                    .add_destination_to_index(push_index, *dst);
            }
        }

        // すべての辺の向きが逆方向であるグラフの隣接リストを作成する
        for dst in 0..self.adjacency_list.len() {
            self.inverse_adjacency_list.resize(self.modules.len(), Vec::<usize>::new());

            for from_id in &self.adjacency_list[dst] {
                self.inverse_adjacency_list[*from_id].push(dst);
            }
        }
    }
}
