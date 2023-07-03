// layers.h: Contains facilities for constructing individual layers of neural networks 

#pragma once
#include <cmath>
#include <Eigen/Core>
#include "../utilities/types.h"
#include "../utilities/traits_concepts.h"
#include "../utilities/softmax.h"

namespace Neural {
	/*
	* @brief: Encapsulates neural net linear layer as a self-contained unit
	* 
	* Uses CRTP pattern for further specialization. Derived class must implement member
	* functions `float activate(float)` and `float differentiate(float)`. The latter should
	* be the derivative of the former.
	* 
	* @tparam EigenType: Must be `Eigen::T<float, Eigen::Dynamic, Eigen::Dynamic>`
	* where `T` is `Array` or `Matrix`
	* @tparam Impl: Derived class implementation (for CRTP)
	*/
	template <typename EigenType, template <typename> class Impl>
	class LinearLayer {
	public:
		auto feedForward(const MatrixX_RowMajor_Ref<float>& inputs) {
			auto aug_inputs = augmentOne(inputs);

			auto signals = aug_inputs * weights;
			auto outputs = signals.unaryExpr([this](float f) 
										     { return this->crtp_handle->activate(f); });

			return std::make_pair(MatOrArray<EigenType>::eval(signals),
								  MatOrArray<EigenType>::eval(outputs));
		}

		// To be used if instance is a hidden layer
		auto backPropagate(const ArrayX_RowMajor_Ref<float>& signals,
						   const ArrayX_RowMajor_Ref<float>& tgradient) {
				auto diff_signals = signals.unaryExpr([this](float f)
													  { return this->crtp_handle->differentiate(f); });

				auto gradient = diff_signals * tgradient;
				auto new_tgradient = transformGradient(diff_signals);

				return std::make_pair(MatOrArray<EigenType>::eval(gradient), new_tgradient);
			}
		
		// To be used if instance is output layer. Corrects @gradient in accordance to the
		// layer's activation function.
		auto seedBackProp(const	ArrayX_RowMajor_Ref<float>& signals,
						  const ArrayX_RowMajor_Ref<float>& gradient) {
			auto diff_signals = signals.unaryExpr([this](float f)
														{ return this->crtp_handle->differentiate(f); });
			auto corrected_gradient = diff_signals * gradient;
			auto tgradient = transformGradient(corrected_gradient);

			return std::make_pair(MatOrArray<EigenType>::eval(corrected_gradient), 
								  MatOrArray<EigenType>::eval(tgradient));
		}
		
		// Updates member `weights` using gradient descent
		void updateWeights(const MatrixX_RowMajor_Ref<float>& inputs, const MatrixX_RowMajor_Ref<float>& gradient, float lr) {
			auto aug_inputs = augmentOne(inputs);
			auto step = lr * (aug_inputs.transpose() * gradient);

			weights -= step;

			return;
		}

	protected:
		LinearLayer(Eigen::Index in_dim, Eigen::Index out_dim, float max_weight,
					int seed = 42) : in_dim(in_dim), out_dim(out_dim), max_weight(max_weight),
									 crtp_handle(static_cast<Impl<EigenType>*>(this))
		{
			std::srand(seed);
			weights = (max_weight * MatrixX_RowMajor<float>::Random(in_dim + 1, out_dim)).eval();
		}
		
		// Used in backpropagation step
		auto transformGradient(const MatrixX_RowMajor_Ref<float>& gradient) {
			return MatOrArray<EigenType>::eval(gradient * weights(Eigen::seq(0, Eigen::last-1), Eigen::all).transpose());
		}

		// Adds column of ones (bias column)
		auto augmentOne(const MatrixX_RowMajor_Ref<float>& to_augment) {
			Eigen::Index num_rows = to_augment.rows();

			MatrixX_RowMajor<float> augmented(num_rows, in_dim + 1);
			
			MatrixX_RowMajor<float> bias_col(num_rows, 1);
			bias_col.setOnes();

			augmented << to_augment, bias_col;

			return augmented.eval();
		}

		MatrixX_RowMajor<float> weights;

		Eigen::Index in_dim;
		Eigen::Index out_dim;

		float max_weight;


	private:
		Impl<EigenType>* crtp_handle;
	};

	/*
	* @brief Implements simplest linear layer (no activation). Inherits from class PlainLinearLayer,
	* using CRTP pattern.
	*
	*/
	template <typename EigenType>
	class PlainLinearLayer : public LinearLayer<EigenType, PlainLinearLayer<EigenType>> {
	public:
		PlainLinearLayer(Eigen::Index in_dim, Eigen::Index out_dim, float max_weight,
						 int seed = 42) : LinearLayer<EigenType, PlainLinearLayer<EigenType>>(in_dim, out_dim,
																							  max_weight, seed) {}
		float activate(float f) {
			return f;
		}

		float differentiate(float f) {
			return 1.0;
		}
	};
}
