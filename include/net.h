// net.h: Contains facilities for the construction of feedforward neural networks

#pragma once
#include <tuple>
#include <functional>
#include <Eigen/Core>
#include "../utilities/types.h"
#include "../include/layers.h"
#include "../include/loss.h"

namespace Neural {
	/*
	* @brief: Encapsulates a feedforward neural network.
	*
	* Uses CRTP for further specialization (e.g. loss function)
	* 
	* @tparam EigenType_1: Must be `Eigen::T<float, Eigen::Dynamic, Eigen::Dynamic>` where `T` is`Matrix
    * 					   or Array
    * @tparam EigenType_2: Must be `Eigen::T<bool, Eigen::Dynamic, Eigen::Dynamic>` where `T` is`Matrix
    * 					   or Array (`T` must be the same in `EigenType_2` and `EigenType_1`)
    * @tparam LayerType: Class of the output layer. Expected to have member functions
    * @tparam Impl: Class of the dervied class (for CRTP). Expected to have member variable `T loss` and
	* 				member function `T evaluate(EigenType_1, EigenType_2)` where `T` can vary.
	* `  
    * `std::pair<EigenType_1, EigenType_1> feedForward(EigenType_1)`,
    * `std::pair<EigenType_1, EigenType_1> seedBackProp(EigenType_1, EigenType_1)` and 
    * `void updateWeights(EigenType_1, EigenType_1, float)`
	*/
	template <typename EigenType_1, typename EigenType_2, typename LayerType, 
			  template <typename, typename, typename> class Impl>
	class FeedFwdNN {
	public:
		/*
		* @brief: Trains neural network. Will update weights of every constituent layer.
		*
		* @param lr: The learning rate
		* @param curr_inputs: Inputs to train on	
		* @param curr_hot_labels: One-hot-shot labels to train on
		* @return: The loss as defined by the derived class implementation
		*/
		auto train(float lr, const EigenType_1& curr_inputs, const EigenType_2& curr_one_hot_labels) {
			auto tup = fwdPass(curr_inputs, curr_one_hot_labels, true);
			auto gradient_vec = bwdPass(std::get<0>(tup), std::get<1>(tup), std::get<2>(tup));
			
			std::vector<EigenType_1> outputs_vec;
			for(auto& pair: std::get<0>(tup)) {
				outputs_vec.push_back(pair.second);
			}

			updateNetwork(outputs_vec, gradient_vec, lr);
			
			return crtp_handle->loss;
		}

		// Overloaded version, uses members `inputs` and `one_hot_labels` as default
		auto train(float lr) {
			return train(lr, inputs, one_hot_labels);
		}

		/*
		* @brief: Tests neural network
		*
		* @param curr_inputs: Inputs to train on	
		* @param curr_hot_labels: One-hot-shot labels to train on
		* @return: The loss as defined by the derived class implementation
		*/
		auto test(const EigenType_1& curr_inputs, const EigenType_2& curr_one_hot_labels) {
			auto tup = fwdPass(curr_inputs, curr_one_hot_labels, false);
			
			return std::get<3>(tup);
		}

		/*
		* @brief: Adds a hidden layer to network. 
		*
		* @tparam LayerType_other: Class of hidden layer to be added. Expected to implement functions`
		* 						   `std::pair<EigenType_1, EigenType_1> feedForward(EigenType_1)`, 
		* 						   `std::pair<EigenType_1, EigenType_1> backPropagate(EigenType_1, EigenType_1)` and
		* 						   `void updateWeights(EigenType_1, EigenType_1, float)`
		* 
		* @param layer: Obj to be added as hidden layer. Must be modifiable.
		*/
		template<typename LayerType_other>
		void pushLayer(LayerType_other& layer) {
			auto feedforward_lambda = [&layer](const EigenType_1& layer_inputs) 
									  { return layer.feedForward(layer_inputs); };
			auto backprop_lambda = [&layer](const EigenType_1& signals, const EigenType_1& next_vec) 
								   { return layer.backPropagate(signals, next_vec); };
			auto update_lambda = [&layer](const EigenType_1& outputs, const EigenType_1& gradient, float lr)
								 { return layer.updateWeights(outputs, gradient, lr); };

			feedforward_funcs.push_back(feedforward_lambda);
			backprop_funcs.push_back(backprop_lambda);
			update_funcs.push_back(update_lambda);
		}

		void popLayer() {
			feedforward_funcs.pop_back();
			backprop_funcs.pop_back();
			update_funcs.pop_back();
		}
		
	protected:
		/* 
		 * @brief: Constructor
		 * 
		 * @param inputs: Default inputs to train on
		 * @param one_hot_labels: Default (one-hot-shot encoded) labels to train on
		 * @param output_layer: Output layer object. Must be modifiable (not const).
		 */
		FeedFwdNN(const EigenType_1& inputs, const EigenType_2& one_hot_labels, LayerType& output_layer) : 
				  inputs(inputs), 
				  one_hot_labels(one_hot_labels),

				  output_feedforward([&output_layer](const EigenType_1& layer_inputs)
			 						{ return output_layer.feedForward(layer_inputs); }),
				  output_seedbackprop([&output_layer](const EigenType_1& signals, const EigenType_1& gradient)
				  					{ return output_layer.seedBackProp(signals, gradient); }),
				  output_update([&output_layer](const EigenType_1& outputs, const EigenType_1& gradient, float lr)
				  			 { return output_layer.updateWeights(outputs, gradient, lr); }),
	   		     
				  crtp_handle(static_cast<Impl<EigenType_1, EigenType_2, LayerType>*>(this))
		{
				static_assert((std::is_same_v<MatrixX_RowMajor<float>, MatrixX_RowMajor<float>>
							   && std::is_same_v<MatrixX_RowMajor<bool>, MatrixX_RowMajor<bool>>) || 
							  (std::is_same_v<MatrixX_RowMajor<float>, ArrayX_RowMajor<float>>
							   && std::is_same_v<MatrixX_RowMajor<bool>, ArrayX_RowMajor<bool>>));
		}

		// Will call `feedForward` function on every constituent layer to perform forward pass.
		auto fwdPass(const EigenType_1& curr_inputs,
					 const EigenType_2& curr_one_hot_labels,
					 bool update_loss = false) {
			std::pair<EigenType_1, EigenType_1> signals_outputs;
			std::vector<std::pair<EigenType_1, EigenType_1>> signals_outputs_vec; 
			
			auto next_inputs = curr_inputs;
			for (auto& func : feedforward_funcs) {
				signals_outputs = func(next_inputs);
				next_inputs = signals_outputs.first;
				signals_outputs_vec.push_back(signals_outputs);
			}
			
			auto final_signals_outputs = output_feedforward(next_inputs);
			auto final_signals = final_signals_outputs.first;
			auto final_outputs = final_signals_outputs.second;
			
			auto entropy_gradient = crtp_handle->evaluate(final_outputs, curr_one_hot_labels);
			if (update_loss) {
				crtp_handle->loss = entropy_gradient.first;
			}
			auto pre_gradient = entropy_gradient.second;
			
			return std::make_tuple(signals_outputs_vec, final_signals, pre_gradient, entropy_gradient.first);
		}

		// Will call `seedBackProp` function of output layer and `backPropagate` function of every 
		// hidden layer to perform backpropagation step.
		auto bwdPass(const std::vector<std::pair<EigenType_1, EigenType_1>>& signals_outputs_vec, 
					 const EigenType_1& final_signals, 
					 const EigenType_1& pre_gradient)
		{	
			auto signals = final_signals;

			auto gradient_tgradient = output_seedbackprop(signals, pre_gradient);
			auto gradient = gradient_tgradient.first;
			auto tgradient = gradient_tgradient.second;
		
			std::vector<EigenType_1> gradient_vec = { gradient };
			for (int i = signals_outputs_vec.size(); i > 0; i--) {
				signals = signals_outputs_vec[i-1].first;

				gradient_tgradient = (backprop_funcs[i-1])(signals, tgradient);
				gradient = gradient_tgradient.first;
				tgradient = gradient_tgradient.second;

				gradient_vec.push_back(gradient);
			}

			return gradient_vec;
		}

		// Will call `updateWeights` function of every constituent layer to update the weights of the network.
		void updateNetwork(const std::vector<EigenType_1>& outputs_vec, const std::vector<EigenType_1>& gradient_vec, float lr) {
			if (lr < 0) {
				throw std::invalid_argument("received negative value for learning rate @lr");
			}

			update_funcs.push_back(output_update);

			update_funcs[0](inputs, gradient_vec[gradient_vec.size() - 1], lr);
			for(int i = 0 ; i < outputs_vec.size() ; i++) {
				update_funcs[i + 1](outputs_vec[i], gradient_vec[gradient_vec.size() - 2 - i], lr);
			}

			update_funcs.pop_back();
		}

		const EigenType_1 inputs;
		const EigenType_2 one_hot_labels;

		const std::function<std::pair<EigenType_1,
								EigenType_1>(const EigenType_1&)> output_feedforward;
		const std::function<std::pair<EigenType_1,
								EigenType_1>(const EigenType_1&, const EigenType_1&)> output_seedbackprop;
		const std::function<void(const EigenType_1&, const EigenType_1&, float)> output_update;

		std::vector<std::function<std::pair<EigenType_1,
										    EigenType_1>(const EigenType_1&)>> feedforward_funcs;
		std::vector<std::function<std::pair<EigenType_1,
											EigenType_1>(const EigenType_1&, const EigenType_1&)>> backprop_funcs;
		std::vector<std::function<void(const EigenType_1&, const EigenType_1&, float)>> update_funcs;

	private:
		Impl<EigenType_1, EigenType_2, LayerType>* crtp_handle;
};

/*
 * @brief: Implementation of categorical cross entropy neural network, derived from class `FeedFwdNN` 
 * 		   using CRTP pattern. 
*/
template<typename EigenType_1, typename EigenType_2, typename LayerType>
class MultiClassNN : public FeedFwdNN<EigenType_1, EigenType_2, LayerType, MultiClassNN<EigenType_1, EigenType_2, LayerType>> {
public:
	MultiClassNN(const EigenType_1& inputs, const EigenType_2& one_hot_labels, LayerType& output_layer) :
		FeedFwdNN<EigenType_1, EigenType_2, LayerType, MultiClassNN<EigenType_1, EigenType_2, LayerType>>(inputs, one_hot_labels, output_layer) {}

	auto evaluate(const EigenType_1& outputs, const EigenType_2& one_hot_labels) {
		auto entropy_gradient = Neural::softMaxLoss(outputs, one_hot_labels);
		return entropy_gradient;
	}
	
	std::pair<float, float> loss;
};
}
