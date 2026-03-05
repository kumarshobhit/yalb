#include "lbm_d2q9.h"

#include <gtest/gtest.h>

#include <utility>

namespace {

constexpr int Nx = 15;
constexpr int Ny = 10;

double total_mass(const lbm_d2q9::DistributionView& f) {
    const auto host_f = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f);
    double mass = 0.0;
    for (int x = 0; x < Nx; ++x) {
        for (int y = 0; y < Ny; ++y) {
            for (int direction = 0; direction < lbm_d2q9::Q; ++direction) {
                mass += host_f(x, y, direction);
            }
        }
    }
    return mass;
}

}  // namespace

TEST(LbmD2Q9Test, InitializeSinglePacketSetsOnlyRequestedPopulation) {
    lbm_d2q9::DistributionView f("f", Nx, Ny, lbm_d2q9::Q);
    lbm_d2q9::initialize_single_packet(f, 5, 5, 1, 1.0);

    const auto host_f = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f);
    EXPECT_DOUBLE_EQ(total_mass(f), 1.0);

    for (int x = 0; x < Nx; ++x) {
        for (int y = 0; y < Ny; ++y) {
            for (int direction = 0; direction < lbm_d2q9::Q; ++direction) {
                const double expected = (x == 5 && y == 5 && direction == 1) ? 1.0 : 0.0;
                EXPECT_DOUBLE_EQ(host_f(x, y, direction), expected);
            }
        }
    }
}

TEST(LbmD2Q9Test, ComputeDensityMatchesOccupiedCell) {
    lbm_d2q9::DistributionView f("f", Nx, Ny, lbm_d2q9::Q);
    lbm_d2q9::DensityView rho("rho", Nx, Ny);

    lbm_d2q9::initialize_single_packet(f, 5, 5, 1, 1.0);
    lbm_d2q9::compute_density(f, rho, Nx, Ny);

    const auto host_rho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rho);
    for (int x = 0; x < Nx; ++x) {
        for (int y = 0; y < Ny; ++y) {
            const double expected = (x == 5 && y == 5) ? 1.0 : 0.0;
            EXPECT_DOUBLE_EQ(host_rho(x, y), expected);
        }
    }
}

TEST(LbmD2Q9Test, ComputeVelocityMatchesEastMovingPacket) {
    lbm_d2q9::DistributionView f("f", Nx, Ny, lbm_d2q9::Q);
    lbm_d2q9::DensityView rho("rho", Nx, Ny);
    lbm_d2q9::VelocityView u("u", Nx, Ny, 2);

    lbm_d2q9::initialize_single_packet(f, 5, 5, 1, 1.0);
    lbm_d2q9::compute_density(f, rho, Nx, Ny);
    lbm_d2q9::compute_velocity(f, rho, u, Nx, Ny);

    const auto host_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u);
    EXPECT_DOUBLE_EQ(host_u(5, 5, 0), 1.0);
    EXPECT_DOUBLE_EQ(host_u(5, 5, 1), 0.0);
}

TEST(LbmD2Q9Test, StreamingMovesPacketOneCellEast) {
    lbm_d2q9::DistributionView f("f", Nx, Ny, lbm_d2q9::Q);
    lbm_d2q9::DistributionView f_next("f_next", Nx, Ny, lbm_d2q9::Q);

    lbm_d2q9::initialize_single_packet(f, 5, 5, 1, 1.0);
    lbm_d2q9::streaming(f, f_next, Nx, Ny);

    const auto host_f_next = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f_next);
    EXPECT_DOUBLE_EQ(host_f_next(6, 5, 1), 1.0);
}

TEST(LbmD2Q9Test, StreamingWrapsAcrossPeriodicBoundary) {
    lbm_d2q9::DistributionView f("f", Nx, Ny, lbm_d2q9::Q);
    lbm_d2q9::DistributionView f_next("f_next", Nx, Ny, lbm_d2q9::Q);

    lbm_d2q9::initialize_single_packet(f, Nx - 1, 5, 1, 1.0);
    lbm_d2q9::streaming(f, f_next, Nx, Ny);

    const auto host_f_next = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f_next);
    EXPECT_DOUBLE_EQ(host_f_next(0, 5, 1), 1.0);
}

TEST(LbmD2Q9Test, StreamingConservesMass) {
    lbm_d2q9::DistributionView f("f", Nx, Ny, lbm_d2q9::Q);
    lbm_d2q9::DistributionView f_next("f_next", Nx, Ny, lbm_d2q9::Q);

    lbm_d2q9::initialize_single_packet(f, 5, 5, 1, 1.0);

    for (int step = 0; step < 10; ++step) {
        lbm_d2q9::streaming(f, f_next, Nx, Ny);
        Kokkos::fence();
        std::swap(f, f_next);
    }

    EXPECT_DOUBLE_EQ(total_mass(f), 1.0);
}
