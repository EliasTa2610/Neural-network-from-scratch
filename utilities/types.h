// types.h: Shorthands for commonly used types in repo

#pragma once
#include<Eigen/Core>

template<typename Scalar>
using MatrixX_RowMajor = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template<typename Scalar>
using MatrixX_RowMajor_Ref = Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

template<typename Scalar>
using MatrixX_ColMajor = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

template<typename Scalar>
using MatRowX = Eigen::Matrix<Scalar, 1, Eigen::Dynamic, Eigen::RowMajor>;

template<typename Scalar>
using MatColX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Eigen::ColMajor>;

template<typename Scalar>
using ArrayX_RowMajor = Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template<typename Scalar>
using ArrayX_RowMajor_Ref = Eigen::Ref<const Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

template<typename Scalar>
using ArrayX_ColMajor = Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

template<typename Scalar>
using ArrRowX = Eigen::Array<Scalar, 1, Eigen::Dynamic, Eigen::RowMajor>;

template<typename Scalar>
using ArrColX = Eigen::Array<Scalar, Eigen::Dynamic, 1, Eigen::ColMajor>;
