#!/bin/sh
# if duplicate-definition error was occured,
# rm -rf include build

GCC_COMMAND="mpicc"
GXX_COMMAND="mpic++"

mkdir ./build
cd ./build
cmake -G "Unix Makefiles" -D CMAKE_C_COMPILER=$GCC_COMMAND -D CMAKE_CXX_COMPILER=$GXX_COMMAND -D CMAKE_BUILD_TYPE=Release ..
make -j 40
make python
cd ../

