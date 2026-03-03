#include "lbm_d2q9.h"

#include <gtest/gtest.h>

#include <utility>

namespace {

constexpr int Nx = 9;
constexpr int Ny = 7;

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

TEST(LbmCollisionTest, EquilibriumAtRestMatchesD2Q9Weights) {
    EXPECT_DOUBLE_EQ(lbm_d2q9::equilibrium_population(1.0, 0.0, 0.0, 0), 4.0 / 9.0);
    EXPECT_DOUBLE_EQ(lbm_d2q9::equilibrium_population(1.0, 0.0, 0.0, 1), 1.0 / 9.0);
    EXPECT_DOUBLE_EQ(lbm_d2q9::equilibrium_population(1.0, 0.0, 0.0, 5), 1.0 / 36.0);
}

TEST(LbmCollisionTest, InitializeFromMacroscopicFieldsRecoversRhoAndVelocity) {
    lbm_d2q9::DensityView rho("rho", Nx, Ny);
    lbm_d2q9::VelocityView u("u", Nx, Ny, 2);
    lbm_d2q9::DistributionView f("f", Nx, Ny, lbm_d2q9::Q);
    lbm_d2q9::DensityView rho_recovered("rho_recovered", Nx, Ny);
    lbm_d2q9::VelocityView u_recovered("u_recovered", Nx, Ny, 2);

    Kokkos::parallel_for(
        "set_macroscopic_fields",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
        KOKKOS_LAMBDA(const int x, const int y) {
            rho(x, y) = 0.2;
            u(x, y, 0) = 0.03;
            u(x, y, 1) = -0.02;
        });

    lbm_d2q9::initialize_from_macroscopic_fields(rho, u, f, Nx, Ny);
    lbm_d2q9::compute_density(f, rho_recovered, Nx, Ny);
    lbm_d2q9::compute_velocity(f, rho_recovered, u_recovered, Nx, Ny);

    const auto host_rho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rho_recovered);
    const auto host_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u_recovered);

    EXPECT_NEAR(host_rho(2, 3), 0.2, 1e-12);
    EXPECT_NEAR(host_u(2, 3, 0), 0.03, 1e-12);
    EXPECT_NEAR(host_u(2, 3, 1), -0.02, 1e-12);
}

TEST(LbmCollisionTest, CollisionWithOmegaOneReturnsCellToEquilibriumForSameMoments) {
    lbm_d2q9::DensityView rho("rho", Nx, Ny);
    lbm_d2q9::VelocityView u("u", Nx, Ny, 2);
    lbm_d2q9::DistributionView f("f", Nx, Ny, lbm_d2q9::Q);

    Kokkos::parallel_for(
        "set_equilibrium_macroscopic_fields",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
        KOKKOS_LAMBDA(const int x, const int y) {
            rho(x, y) = 0.2;
            u(x, y, 0) = 0.0;
            u(x, y, 1) = 0.0;
        });

    lbm_d2q9::initialize_from_macroscopic_fields(rho, u, f, Nx, Ny);

    auto host_f = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f);
    host_f(4, 3, 1) += 0.01;
    host_f(4, 3, 3) += 0.01;
    host_f(4, 3, 2) -= 0.01;
    host_f(4, 3, 4) -= 0.01;
    Kokkos::deep_copy(f, host_f);

    lbm_d2q9::compute_density(f, rho, Nx, Ny);
    lbm_d2q9::compute_velocity(f, rho, u, Nx, Ny);
    lbm_d2q9::collide_bgk(f, rho, u, 1.0, Nx, Ny);

    const auto collided = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f);
    for (int direction = 0; direction < lbm_d2q9::Q; ++direction) {
        EXPECT_NEAR(collided(4, 3, direction), lbm_d2q9::equilibrium_population(0.2, 0.0, 0.0, direction), 1e-12);
    }
}

TEST(LbmCollisionTest, StreamingAndCollisionConserveMass) {
    lbm_d2q9::DensityView rho("rho", Nx, Ny);
    lbm_d2q9::VelocityView u("u", Nx, Ny, 2);
    lbm_d2q9::DistributionView f("f", Nx, Ny, lbm_d2q9::Q);
    lbm_d2q9::DistributionView f_next("f_next", Nx, Ny, lbm_d2q9::Q);

    Kokkos::parallel_for(
        "set_bump_for_mass_test",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
        KOKKOS_LAMBDA(const int x, const int y) {
            rho(x, y) = 0.1;
            if (x == Nx / 2 && y == Ny / 2) {
                rho(x, y) = 0.15;
            }
            u(x, y, 0) = 0.02;
            u(x, y, 1) = 0.0;
        });

    lbm_d2q9::initialize_from_macroscopic_fields(rho, u, f, Nx, Ny);
    const double initial_mass = total_mass(f);

    for (int step = 0; step < 5; ++step) {
        lbm_d2q9::streaming(f, f_next, Nx, Ny);
        lbm_d2q9::compute_density(f_next, rho, Nx, Ny);
        lbm_d2q9::compute_velocity(f_next, rho, u, Nx, Ny);
        lbm_d2q9::collide_bgk(f_next, rho, u, 1.2, Nx, Ny);
        Kokkos::fence();
        std::swap(f, f_next);
    }

    EXPECT_NEAR(total_mass(f), initial_mass, 1e-12);
}
