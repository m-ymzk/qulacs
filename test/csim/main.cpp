#include <mpi.h>
#include <gtest/gtest.h>

GTEST_API_ int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);

    MPI_Init(&argc, &argv);

    int ret = RUN_ALL_TESTS();

    MPI_Finalize();

    return ret;
}
