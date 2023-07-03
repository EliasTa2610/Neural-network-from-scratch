// trait_concepts.h: Contains useful traits and concepts

#pragma once
#include <Eigen/Core>
#include "types.h"

template<typename Derived>
struct EigenTraits {
	using Scalar = typename Derived::Scalar; 
	constexpr static auto NumRows = Derived::RowsAtCompileTime;
	constexpr static auto NumCols = Derived::ColsAtCompileTime;
	//constexpr static auto Options = Derived::Options;
	constexpr static auto MaxRows = Derived::MaxRowsAtCompileTime;
    constexpr static auto MaxCols = Derived::MaxColsAtCompileTime;
};

template<typename T>
struct MatOrArray {};

template<typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
struct MatOrArray<Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>> {
	template<typename MatDerived>
	static auto eval(const Eigen::MatrixBase<MatDerived>& mat) {
		return mat.eval();
	}

	template<typename ArrDerived>
	static auto eval(const Eigen::ArrayBase<ArrDerived>& arr) {
		return arr.eval().matrix().eval();
	}
};

template<typename Derived>
struct MatOrArray<const Eigen::MatrixBase<Derived>> : public MatOrArray<Eigen::MatrixBase<Derived>> {};

template<typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
struct MatOrArray<Eigen::Array<Scalar, Rows, Cols, Options, MaxRows, MaxCols>> {
	template<typename MatDerived>
	static auto eval(const Eigen::MatrixBase<MatDerived>& mat) {
		return mat.eval().array().eval();
	}

	template<typename ArrDerived>
	static auto eval(const Eigen::ArrayBase<ArrDerived>& arr) {
		return arr.eval();
	}
};

template<typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
struct MatOrArray<const Eigen::Array<Scalar, Rows, Cols, Options, MaxRows, MaxCols>> : public MatOrArray<Eigen::Array<Scalar, Rows, Cols, Options, MaxRows, MaxCols>> {};

template<typename Derived>
concept IsEigenFloat = std::is_floating_point<typename Derived::Scalar>::value;

template<typename Derived>
concept IsEigenInt = std::is_same<typename Derived::Scalar, int>::value;

template<typename Derived>
concept IsEigenBool = std::is_same<typename Derived::Scalar, bool>::value;

