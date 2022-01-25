#!/bin/bash

rm -f _log_regression_n*
mpirun -n 2 ../bin/cppsim_test --gtest_filter=*multicpu* 2>&1 |tee -a _log_regression_n2
mpirun -n 4 ../bin/cppsim_test --gtest_filter=*multicpu* 2>&1 |tee -a _log_regression_n4
mpirun -n 8 ../bin/cppsim_test --gtest_filter=*multicpu* 2>&1 |tee -a _log_regression_n8
mpirun -n 16 ../bin/cppsim_test --gtest_filter=*multicpu* 2>&1 |tee -a _log_regression_n16
mpirun -n 32 ../bin/cppsim_test --gtest_filter=*multicpu* 2>&1 |tee -a _log_regression_n32

echo Num of FAIL, PASSED, log_file
for f in _log_regression_n*; do
   	fn=`grep FAIL $f | wc -l`
	ps=`grep PASSED $f | wc -l`
	echo $fn $ps $f
done
