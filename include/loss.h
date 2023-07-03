// loss.h: Contains facilities implementing loss functions for use in neural networks

#pragma once
#include <Eigen/Core>
#include "../utilities/types.h"
#include "../utilities/paral.h"
#include "../utilities/softmax.h"
#include "../include/labels.h"

namespace Neural {
    /*
    * @brief: Implements categorical cross-entropy (softmax) loss
    *
    * @param outputs: Outputs of output layer of network
    * @param one_hot_labels: One-hot-shot encoded labels
    * @return: `std::pair<std::pair<float, float>, MatrixX_RowMajor<float>>` where
    *          floats are cross-entropy and misclassification resp. and 
               `MatrixX_RowMajor<float>` is the gradient of the categorical cross-entropy
               loss wrt the incoming signals (assuming no activation function)
    */
    static auto _softMaxLoss(const Eigen::Ref<const MatrixX_RowMajor<float>>& outputs,
                             const Eigen::Ref<const MatrixX_RowMajor<bool>>& one_hot_labels) {
        auto num_rows = one_hot_labels.rows();
        auto& labels_float = one_hot_labels.cast<float>();
        auto softmaxed = softMax(outputs, Ax::One);
        
        auto probs = (softmaxed.array() * labels_float.array()).rowwise().sum();
        auto logits = probs.unaryExpr(std::ref(std::logf));
        float cross_entropy = (-1.0 / (float)num_rows) * logits.sum();
        
        ArrayX_RowMajor<int> max_col = ArrayX_RowMajor<int>(num_rows, 1);
        auto indices_labels = Labels::toIndicesLabels(one_hot_labels);
        rangeParExec(
            num_rows,
            [&](int& row_number) {
                int i;
                softmaxed.row(row_number).maxCoeff(&i);
                max_col(row_number) = i;
            }
        );
        float misclas = (1.0 / (float)num_rows)*((max_col != indices_labels.array()).cast<float>().sum());

        auto gradient = (1.0 / num_rows) * (softmaxed - labels_float);

        return std::make_pair(std::make_pair(cross_entropy, misclas),  gradient.eval());
    }

    // Versions surfaced to client; honor `Eigen::Array` or `Eigen::Matrix` depending
    // on input

    template<typename Derived_1, typename Derived_2>
    auto softMaxLoss(const Eigen::MatrixBase<Derived_1>& outputs, 
                     const Eigen::MatrixBase<Derived_2>& one_hot_labels) {
        return _softMaxLoss(outputs, one_hot_labels);
    }

    template<typename Derived_1, typename Derived_2>
    auto softMaxLoss(const Eigen::ArrayBase<Derived_1>& outputs,
                     const Eigen::ArrayBase<Derived_2>& one_hot_labels) {
        auto pair = _softMaxLoss(outputs, one_hot_labels);
        return std::make_pair(pair.first, pair.second.array().eval());
    }
}
