// paral.h: Contains facility for computing vectorizable operation in parallel

#pragma once
#include <algorithm>
#include <execution>
#include <Eigen/Core>
#include <utilities/types.h>

template<typename UnaryFunction>
void rangeParExec(Eigen::Index max, const UnaryFunction& func) {
	MatColX<int> range(max);
	range.setLinSpaced(0, (int) max);

	std::for_each(
		std::execution::par,
		range.begin(),
		range.end(),
		func
	);
}
