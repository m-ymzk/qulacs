#!/bin/sh

set -eux

GCC_COMMAND=${C_COMPILER:-"gcc"}
GXX_COMMAND=${CXX_COMPILER:-"g++"}
USE_GPU="${USE_GPU:-No}"
USE_MPI="${USE_MPI:-No}"
USE_TEST="${USE_TEST:-No}"

mkdir -p ./build
cd ./build
if [ "${QULACS_OPT_FLAGS:-"__UNSET__"}" = "__UNSET__" ]; then
    DEFINE_OPT_FLAGS=""
else
    DEFINE_OPT_FLAGS="-D OPT_FLAGS=\"${QULACS_OPT_FLAGS}\""
fi

cmake -G "Unix Makefiles" -D CMAKE_C_COMPILER=$GCC_COMMAND -D CMAKE_CXX_COMPILER=$GXX_COMMAND \
      $DEFINE_OPT_FLAGS \
      -D CMAKE_BUILD_TYPE=Release -D USE_GPU=$USE_GPU -D USE_TEST=$USE_TEST -D USE_MPI=$USE_MPI \
      ..

make -j $(nproc)
cd ../
