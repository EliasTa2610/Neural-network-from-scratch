// labels.h: Contains facilities for dealing with indices and one-hot-shot labels 

#pragma once
#include <stdexcept>
#include <Eigen/Core>
#include "../utilities/types.h"
#include "../utilities/traits_concepts.h"
#include "../utilities/paral.h"

namespace Labels {
    // Internal implementations

    /**
    * @brief Converts bool, dynamic, row-major <Eigen::Matrix> obj (or castable thereof)
    *        @one_hot_labels, representing one-hot-shot labels, to obj representing
    *        indices labels.
    *
    * @param one_hot_labels one-hot-shot encoded labels
    * @return int, dynamic, row-major <Eigen::Matrix> obj containing indices labels
    */
    static auto _toIndicesLabels(const MatrixX_RowMajor_Ref<bool>& one_hot_labels) {
        auto& one_hot_labels_int = one_hot_labels.cast<int>();

        Eigen::Index num_classes = one_hot_labels.cols();
        MatColX<int> indices(num_classes);
        indices.setLinSpaced(0, (int)num_classes);

        auto indices_labels = (one_hot_labels_int * indices).eval();

        return indices_labels;
    }

    /**
    * @brief Converts int, dynamic, row-major <Eigen::Matrix> obj (or castable thereof)
    *        @indices_labels, representing indices labels, to obj representing
    *        one-hot-shot labels. Number of label calsses must be specified through
             * <Eigen::Index> obj @num_classes.
    *
    * @param indices_labels: indices labels
    * @param num_classes: number of label classes
    * @return: bool, dynamic, row-major <Eigen::Matrix> obj representing one-hot-shot
    *          labels
    */
    static auto _toOneHotLabels(const Eigen::Ref<const MatColX<int>>& indices_labels, Eigen::Index num_classes) {
        if (indices_labels.minCoeff() < 0) {
            throw std::invalid_argument("received negative values");
        }
        if (indices_labels.maxCoeff() >= (int)num_classes) {
            throw std::invalid_argument("max value does not match argument `num_classes`");
        }

        Eigen::Index num_rows = indices_labels.rows();
        MatrixX_RowMajor<bool> one_hot_labels;
        one_hot_labels.resize(num_rows, num_classes);
        one_hot_labels.setZero();

        rangeParExec(
            num_rows,
            [&](int& row_number) {
                *(one_hot_labels.row(row_number).begin() + indices_labels[row_number]) = true;
            }
        );

        return one_hot_labels;
    }

    // Versions surfaced to client; honor `Eigen::Array` or `Eigen::Matrix` depending
    // on input

    template<typename Derived>
    auto toIndicesLabels(const Eigen::MatrixBase<Derived>& one_hot_labels) {
        return _toIndicesLabels(one_hot_labels);
    }

    template<typename Derived>
    auto toIndicesLabels(const Eigen::ArrayBase<Derived>& one_hot_labels) {
        return _toIndicesLabels(one_hot_labels).array().eval();
    }

    template<typename Derived>
    auto toOneHotLabels(const Eigen::MatrixBase<Derived>& indices_labels, Eigen::Index num_classes) {
        return _toOneHotLabels(indices_labels, num_classes);
    }

    template<typename Derived>
    auto toOneHotLabels(const Eigen::ArrayBase<Derived>& indices_labels, Eigen::Index num_classes) {
        return _toOneHotLabels(indices_labels, num_classes).array().eval();
    }
}
