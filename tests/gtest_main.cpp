#include <Kokkos_Core.hpp>

#include "gtest/gtest.h"
GTEST_API_ int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);

    Kokkos::initialize(argc, argv);
    const int result = RUN_ALL_TESTS();
    Kokkos::finalize();

    return result;
}
