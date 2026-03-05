#include "lbm_d2q9.h"

#include <gtest/gtest.h>

#include <cmath>
#include <utility>

namespace {

constexpr int Nx = 16;
constexpr int Ny = 16;

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

TEST(LbmShearWaveTest, TheoreticalViscosityMatchesOmegaOne) {
    EXPECT_NEAR(lbm_d2q9::theoretical_viscosity_from_omega(1.0), 1.0 / 6.0, 1e-12);
}

TEST(LbmShearWaveTest, AnalyticalViscosityCurveMonotonicInOmegaRange) {
    const double nu_06 = lbm_d2q9::theoretical_viscosity_from_omega(0.6);
    const double nu_10 = lbm_d2q9::theoretical_viscosity_from_omega(1.0);
    const double nu_14 = lbm_d2q9::theoretical_viscosity_from_omega(1.4);

    EXPECT_GT(nu_06, nu_10);
    EXPECT_GT(nu_10, nu_14);
    EXPECT_GT(nu_14, 0.0);
}

TEST(LbmShearWaveTest, InitializeShearWaveCreatesUniformDensity) {
    lbm_d2q9::DensityView rho("rho", Nx, Ny);
    lbm_d2q9::VelocityView u("u", Nx, Ny, 2);

    lbm_d2q9::initialize_shear_wave_macroscopic_fields(rho, u, Nx, Ny, 0.2, 0.05);

    const auto host_rho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rho);
    const auto host_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u);

    for (int x = 0; x < Nx; ++x) {
        for (int y = 0; y < Ny; ++y) {
            EXPECT_NEAR(host_rho(x, y), 0.2, 1e-12);
            EXPECT_NEAR(host_u(x, y, 1), 0.0, 1e-12);
        }
    }
}

TEST(LbmShearWaveTest, InitializeShearWaveCreatesCorrectSinusoidalUx) {
    lbm_d2q9::DensityView rho("rho", Nx, Ny);
    lbm_d2q9::VelocityView u("u", Nx, Ny, 2);

    lbm_d2q9::initialize_shear_wave_macroscopic_fields(rho, u, Nx, Ny, 0.2, 0.05);
    const auto host_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u);

    for (int y : {0, 4, 8, 12}) {
        const double expected = 0.05 * std::sin(2.0 * lbm_d2q9::pi * static_cast<double>(y) / static_cast<double>(Ny));
        EXPECT_NEAR(host_u(3, y, 0), expected, 1e-12);
    }
}

TEST(LbmShearWaveTest, InitializeFromMacroscopicFieldsRecoversShearWave) {
    lbm_d2q9::DensityView rho("rho", Nx, Ny);
    lbm_d2q9::VelocityView u("u", Nx, Ny, 2);
    lbm_d2q9::DistributionView f("f", Nx, Ny, lbm_d2q9::Q);
    lbm_d2q9::DensityView rho_recovered("rho_recovered", Nx, Ny);
    lbm_d2q9::VelocityView u_recovered("u_recovered", Nx, Ny, 2);

    lbm_d2q9::initialize_shear_wave_macroscopic_fields(rho, u, Nx, Ny, 0.2, 0.05);
    lbm_d2q9::initialize_from_macroscopic_fields(rho, u, f, Nx, Ny);
    lbm_d2q9::compute_density(f, rho_recovered, Nx, Ny);
    lbm_d2q9::compute_velocity(f, rho_recovered, u_recovered, Nx, Ny);

    const auto host_rho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rho_recovered);
    const auto host_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u_recovered);

    for (int y : {1, 5, 9, 13}) {
        const double expected = 0.05 * std::sin(2.0 * lbm_d2q9::pi * static_cast<double>(y) / static_cast<double>(Ny));
        EXPECT_NEAR(host_rho(2, y), 0.2, 1e-10);
        EXPECT_NEAR(host_u(2, y, 0), expected, 1e-10);
        EXPECT_NEAR(host_u(2, y, 1), 0.0, 1e-10);
    }
}

TEST(LbmShearWaveTest, MeasureShearWaveAmplitudeRecoversKnownAmplitude) {
    lbm_d2q9::DensityView rho("rho", Nx, Ny);
    lbm_d2q9::VelocityView u("u", Nx, Ny, 2);

    lbm_d2q9::initialize_shear_wave_macroscopic_fields(rho, u, Nx, Ny, 0.2, 0.05);
    EXPECT_NEAR(lbm_d2q9::measure_shear_wave_amplitude(u, Nx, Ny), 0.05, 1e-12);
}

TEST(LbmShearWaveTest, ZeroAmplitudeRemainsZero) {
    lbm_d2q9::DensityView rho("rho", Nx, Ny);
    lbm_d2q9::VelocityView u("u", Nx, Ny, 2);
    lbm_d2q9::DistributionView f("f", Nx, Ny, lbm_d2q9::Q);
    lbm_d2q9::DistributionView f_next("f_next", Nx, Ny, lbm_d2q9::Q);

    lbm_d2q9::initialize_shear_wave_macroscopic_fields(rho, u, Nx, Ny, 0.2, 0.0);
    lbm_d2q9::initialize_from_macroscopic_fields(rho, u, f, Nx, Ny);

    for (int step = 0; step < 8; ++step) {
        lbm_d2q9::streaming(f, f_next, Nx, Ny);
        lbm_d2q9::compute_density(f_next, rho, Nx, Ny);
        lbm_d2q9::compute_velocity(f_next, rho, u, Nx, Ny);
        lbm_d2q9::collide_bgk(f_next, rho, u, 1.0, Nx, Ny);
        Kokkos::fence();
        std::swap(f, f_next);
    }

    EXPECT_NEAR(lbm_d2q9::measure_shear_wave_amplitude(u, Nx, Ny), 0.0, 1e-12);
}

TEST(LbmShearWaveTest, ShearWaveSimulationConservesMass) {
    lbm_d2q9::DensityView rho("rho", Nx, Ny);
    lbm_d2q9::VelocityView u("u", Nx, Ny, 2);
    lbm_d2q9::DistributionView f("f", Nx, Ny, lbm_d2q9::Q);
    lbm_d2q9::DistributionView f_next("f_next", Nx, Ny, lbm_d2q9::Q);

    lbm_d2q9::initialize_shear_wave_macroscopic_fields(rho, u, Nx, Ny, 0.2, 0.05);
    lbm_d2q9::initialize_from_macroscopic_fields(rho, u, f, Nx, Ny);
    const double initial_mass = total_mass(f);

    for (int step = 0; step < 8; ++step) {
        lbm_d2q9::streaming(f, f_next, Nx, Ny);
        lbm_d2q9::compute_density(f_next, rho, Nx, Ny);
        lbm_d2q9::compute_velocity(f_next, rho, u, Nx, Ny);
        lbm_d2q9::collide_bgk(f_next, rho, u, 1.0, Nx, Ny);
        Kokkos::fence();
        std::swap(f, f_next);
    }

    EXPECT_NEAR(total_mass(f), initial_mass, 1e-12);
}
