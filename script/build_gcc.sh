#!/bin/sh

set -eux

GCC_COMMAND=${C_COMPILER:-"gcc"}
GXX_COMMAND=${CXX_COMPILER:-"g++"}

USE_GPU=${USE_GPU:-"No"}
USE_MPI=${USE_MPI:-"No"}
USE_TEST=${USE_TEST:-"No"}

CMAKE_OPTIONS="-D CMAKE_C_COMPILER=$GCC_COMMAND -D CMAKE_CXX_COMPILER=$GXX_COMMAND"
CMAKE_OPTIONS="$CMAKE_OPTIONS -D USE_GPU=${USE_GPU} -D USE_MPI=${USE_MPI} -D USE_TEST=${USE_TEST}"
CMAKE_OPTIONS="$CMAKE_OPTIONS -D CMAKE_BUILD_TYPE=Release"

mkdir -p ./build
cd ./build
if [ "${QULACS_OPT_FLAGS:-"__UNSET__"}" = "__UNSET__" ]; then
  cmake -G "Unix Makefiles" $CMAKE_OPTIONS ..
else
  cmake -G "Unix Makefiles" $CMAKE_OPTIONS -D OPT_FLAGS="${QULACS_OPT_FLAGS}" ..
fi
make -j $(nproc)
cd ../
