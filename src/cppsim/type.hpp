#pragma once
#ifndef TYPE_HPP_
#define TYPE_HPP_

#ifndef _MSC_VER
extern "C" {
#include <csim/type.h>
}
#else
#include <csim/type.h>
#endif

#include <complex>
#include <Eigen/Core>
#include <Eigen/Sparse>
typedef std::complex<double> CPPCTYPE;
typedef Eigen::VectorXcd ComplexVector;

// In order to use matrix raw-data without reordering, we use RowMajor as default.
typedef Eigen::Matrix<CPPCTYPE, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ComplexMatrix;
typedef Eigen::SparseMatrix<CPPCTYPE> SparseComplexMatrix;

#ifdef __GNUC__
#if __GNUC__ >= 8
using namespace std::complex_literals;
#endif
#endif

#ifdef _MSC_VER
inline static FILE* popen(const char* command, const char* mode) { return _popen(command, mode); }
inline static void pclose(FILE* fp) { _pclose(fp);  }
#define strncasecmp _strnicmp
#define strcasecmp _stricmp
#endif

#endif
