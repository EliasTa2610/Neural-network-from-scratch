// softmax.h: Implements softmax function

#pragma once
#include <cmath>
#include <functional>
#include <iostream>
#include <Eigen/Core>
#include <utilities/types.h>
#include <utilities/traits_concepts.h>

enum class Ax {Zero, One, None};

auto softMax(const Eigen::Ref<const MatrixX_RowMajor<float>>& input, Ax axis = Ax::None) {
	auto raised = input.unaryExpr(std::ref(std::expf)).eval();
	
	if (axis == Ax::Zero) {
		MatRowX<float> s = raised.colwise().sum();
		auto softmaxed = raised * s.asDiagonal().inverse();

		return softmaxed.eval();
	}
	else if (axis == Ax::One) {
		MatColX<float> s = raised.rowwise().sum();
		auto softmaxed = s.asDiagonal().inverse() * raised;

		return softmaxed.eval();
	}
	else {
		float s = raised.sum();
		auto softmaxed = (1.0 / s) * raised;

		return softmaxed.eval();
	}
}

template<typename Derived>
auto softMax(const Eigen::ArrayBase<Derived>& input, Ax axis = Ax::None) {
	return softmax(input.matrix().eval(), axis).array().eval();
}
