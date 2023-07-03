// input.h : Contains facilities for reading in input stream

#pragma once
#include <string>
#include <istream>
#include <Eigen/Dense>
#include "utilities/types.h"

namespace Input {
	// Efficient resize for amortized O(1) element-wise copies
	template<typename Derived>
	static inline void addRow(Eigen::PlainObjectBase<Derived>& mat_arr, Eigen::Index num_filled_rows) {
		Eigen::Index num_total_rows = mat_arr.rows();
		if (num_filled_rows == num_total_rows) {
			mat_arr.conservativeResize(2 * num_total_rows, mat_arr.cols());
		}
		return;
	}

	/** 
	* @brief: Reads in `istream` @ist into an `Eigen::Array` and returns the latter. 
	*		  Expects @ist to consist of newline-delimited lines, each consisting of 
	*		  the same number of items (item delimiters determined by `std::cout` 
	*		  based on template parameter @Scalar). 
	*
	* @tparam Scalar: type determining the delimeters in each line
	* @param ist: input stream obj
	* @return: `Eigen::Array` obj containing data
	*/
	template<typename Scalar = float>
	auto readData(std::istream& ist) {
		std::string line;
		std::vector<Scalar> single_row;
		ArrayX_RowMajor<Scalar> data;

		std::getline(ist, line);
		std::istringstream line_ist{ line };

		Scalar val;
		while (line_ist >> val) {
			single_row.push_back(val);
		}

		auto num_cols = single_row.size();
		data.resize(1, num_cols);
		auto single_row_eigen = Eigen::Map<ArrColX<Scalar>>(single_row.data(), num_cols);

		data.row(0) = single_row_eigen;

		Eigen::Index row_counter{ 1 };
		while (std::getline(ist, line)) {
			addRow(data, row_counter);

			line_ist = std::istringstream(line);
			for (int i = 0; i < num_cols; i++) {
				line_ist >> single_row[i];
			}

			data.row(row_counter) = single_row_eigen;

			row_counter++;
		}

		data.conservativeResize(row_counter, num_cols);

		return data;
	}
}
