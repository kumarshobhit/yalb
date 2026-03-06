#include "lbm_d2q9.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace {

constexpr int Nx = 24;
constexpr int Ny = 24;

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

double max_speed(const lbm_d2q9::VelocityView& u, int x_begin, int x_end, int y_begin, int y_end) {
    const auto host_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u);
    double speed_max = 0.0;

    for (int x = x_begin; x < x_end; ++x) {
        for (int y = y_begin; y < y_end; ++y) {
            const double ux = host_u(x, y, 0);
            const double uy = host_u(x, y, 1);
            speed_max = std::max(speed_max, std::sqrt(ux * ux + uy * uy));
        }
    }

    return speed_max;
}

double max_abs_uy_on_row(const lbm_d2q9::VelocityView& u, int y_row, int x_begin, int x_end) {
    const auto host_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u);
    double max_uy = 0.0;
    for (int x = x_begin; x < x_end; ++x) {
        max_uy = std::max(max_uy, std::abs(host_u(x, y_row, 1)));
    }
    return max_uy;
}

void run_cavity_steps(
    int steps,
    double omega,
    double rho0,
    double lid_ux,
    double lid_uy,
    lbm_d2q9::DistributionView& f,
    lbm_d2q9::DistributionView& f_next,
    lbm_d2q9::DensityView& rho,
    lbm_d2q9::VelocityView& u) {
    lbm_d2q9::initialize_uniform_macroscopic_fields(rho, u, Nx, Ny, rho0, 0.0, 0.0);
    lbm_d2q9::initialize_from_macroscopic_fields(rho, u, f, Nx, Ny);

    for (int step = 0; step < steps; ++step) {
        lbm_d2q9::stream_with_cavity_boundaries(f, f_next, Nx, Ny, lid_ux, lid_uy, rho0);
        lbm_d2q9::compute_density(f_next, rho, Nx, Ny);
        lbm_d2q9::compute_velocity(f_next, rho, u, Nx, Ny);
        lbm_d2q9::collide_bgk(f_next, rho, u, omega, Nx, Ny);
        Kokkos::fence();
        std::swap(f, f_next);
    }
}

}  // namespace

TEST(LbmCavityTest, OppositeDirectionMappingIsCorrect) {
    constexpr std::array<int, lbm_d2q9::Q> expected = {0, 3, 4, 1, 2, 7, 8, 5, 6};
    for (int direction = 0; direction < lbm_d2q9::Q; ++direction) {
        EXPECT_EQ(lbm_d2q9::opposite(direction), expected[direction]);
    }
}

TEST(LbmCavityTest, StationaryWallsNoLidRemainAtRest) {
    lbm_d2q9::DistributionView f("f", Nx, Ny, lbm_d2q9::Q);
    lbm_d2q9::DistributionView f_next("f_next", Nx, Ny, lbm_d2q9::Q);
    lbm_d2q9::DensityView rho("rho", Nx, Ny);
    lbm_d2q9::VelocityView u("u", Nx, Ny, 2);

    run_cavity_steps(40, 1.0, 1.0, 0.0, 0.0, f, f_next, rho, u);

    EXPECT_LE(max_speed(u, 0, Nx, 0, Ny), 1e-12);
}

TEST(LbmCavityTest, MovingLidGeneratesNonZeroFlow) {
    lbm_d2q9::DistributionView f("f", Nx, Ny, lbm_d2q9::Q);
    lbm_d2q9::DistributionView f_next("f_next", Nx, Ny, lbm_d2q9::Q);
    lbm_d2q9::DensityView rho("rho", Nx, Ny);
    lbm_d2q9::VelocityView u("u", Nx, Ny, 2);

    run_cavity_steps(120, 1.0, 1.0, 0.05, 0.0, f, f_next, rho, u);

    const double interior_speed = max_speed(u, 2, Nx - 2, 2, Ny - 2);
    EXPECT_GT(interior_speed, 1e-6);
}

TEST(LbmCavityTest, MassConservedInClosedCavity) {
    lbm_d2q9::DistributionView f("f", Nx, Ny, lbm_d2q9::Q);
    lbm_d2q9::DistributionView f_next("f_next", Nx, Ny, lbm_d2q9::Q);
    lbm_d2q9::DensityView rho("rho", Nx, Ny);
    lbm_d2q9::VelocityView u("u", Nx, Ny, 2);

    const double rho0 = 1.0;
    lbm_d2q9::initialize_uniform_macroscopic_fields(rho, u, Nx, Ny, rho0, 0.0, 0.0);
    lbm_d2q9::initialize_from_macroscopic_fields(rho, u, f, Nx, Ny);
    const double initial_mass = total_mass(f);

    for (int step = 0; step < 300; ++step) {
        lbm_d2q9::stream_with_cavity_boundaries(f, f_next, Nx, Ny, 0.05, 0.0, rho0);
        lbm_d2q9::compute_density(f_next, rho, Nx, Ny);
        lbm_d2q9::compute_velocity(f_next, rho, u, Nx, Ny);
        lbm_d2q9::collide_bgk(f_next, rho, u, 1.0, Nx, Ny);
        Kokkos::fence();
        std::swap(f, f_next);
    }

    EXPECT_NEAR(total_mass(f), initial_mass, 1e-10);
}

TEST(LbmCavityTest, NoPenetrationAtTopBottomInSteadyingRun) {
    lbm_d2q9::DistributionView f("f", Nx, Ny, lbm_d2q9::Q);
    lbm_d2q9::DistributionView f_next("f_next", Nx, Ny, lbm_d2q9::Q);
    lbm_d2q9::DensityView rho("rho", Nx, Ny);
    lbm_d2q9::VelocityView u("u", Nx, Ny, 2);

    run_cavity_steps(400, 1.0, 1.0, 0.05, 0.0, f, f_next, rho, u);

    EXPECT_LE(max_abs_uy_on_row(u, 0, 2, Nx - 2), 1e-3);
    EXPECT_LE(max_abs_uy_on_row(u, Ny - 1, 2, Nx - 2), 3e-3);
}
