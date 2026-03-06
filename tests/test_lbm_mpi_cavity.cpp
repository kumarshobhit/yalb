#include "lbm_d2q9.h"
#include "lbm_d2q9_mpi.h"

#include <mpi.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <utility>

namespace {

constexpr int Ny = 16;

TEST(LbmMpiCavityTest, DomainDecompositionCoversGlobalXExactly) {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int nx = std::max(32, size + 2);
    const auto decomp = lbm_d2q9::decompose_domain_1d_x(nx, Ny, MPI_COMM_WORLD);

    int sum_local_nx = 0;
    MPI_Allreduce(&decomp.local_nx, &sum_local_nx, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_EQ(sum_local_nx, nx);

    int expected_offset = 0;
    MPI_Exscan(&decomp.local_nx, &expected_offset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        expected_offset = 0;
    }
    EXPECT_EQ(decomp.x_offset, expected_offset);
}

TEST(LbmMpiCavityTest, HaloExchangeTransfersBoundaryColumns) {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int nx = std::max(40, size + 2);
    const auto decomp = lbm_d2q9::decompose_domain_1d_x(nx, Ny, MPI_COMM_WORLD);

    lbm_d2q9::DistributionView f("f", decomp.local_nx + 2, Ny, lbm_d2q9::Q);
    auto host_f = Kokkos::create_mirror_view(f);

    for (int x = 0; x < decomp.local_nx + 2; ++x) {
        for (int y = 0; y < Ny; ++y) {
            for (int d = 0; d < lbm_d2q9::Q; ++d) {
                host_f(x, y, d) = -1.0;
            }
        }
    }

    for (int y = 0; y < Ny; ++y) {
        for (int d = 0; d < lbm_d2q9::Q; ++d) {
            host_f(1, y, d) = 1000.0 * static_cast<double>(rank) + 10.0 * static_cast<double>(y) + static_cast<double>(d);
            host_f(decomp.local_nx, y, d) =
                2000.0 * static_cast<double>(rank) + 10.0 * static_cast<double>(y) + static_cast<double>(d);
        }
    }

    Kokkos::deep_copy(f, host_f);
    lbm_d2q9::exchange_halo_columns(f, decomp, MPI_COMM_WORLD);

    const auto exchanged = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f);

    if (decomp.left_rank != MPI_PROC_NULL) {
        for (int y = 0; y < Ny; ++y) {
            for (int d = 0; d < lbm_d2q9::Q; ++d) {
                const double expected =
                    2000.0 * static_cast<double>(rank - 1) + 10.0 * static_cast<double>(y) + static_cast<double>(d);
                EXPECT_DOUBLE_EQ(exchanged(0, y, d), expected);
            }
        }
    }

    if (decomp.right_rank != MPI_PROC_NULL) {
        for (int y = 0; y < Ny; ++y) {
            for (int d = 0; d < lbm_d2q9::Q; ++d) {
                const double expected =
                    1000.0 * static_cast<double>(rank + 1) + 10.0 * static_cast<double>(y) + static_cast<double>(d);
                EXPECT_DOUBLE_EQ(exchanged(decomp.local_nx + 1, y, d), expected);
            }
        }
    }
}

TEST(LbmMpiCavityTest, MassConservedAcrossRanks) {
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int nx = std::max(64, size + 2);
    const int ny = 32;
    const auto decomp = lbm_d2q9::decompose_domain_1d_x(nx, ny, MPI_COMM_WORLD);

    lbm_d2q9::DistributionView f("f", decomp.local_nx + 2, ny, lbm_d2q9::Q);
    lbm_d2q9::DistributionView f_next("f_next", decomp.local_nx + 2, ny, lbm_d2q9::Q);
    lbm_d2q9::DensityView rho("rho", decomp.local_nx, ny);
    lbm_d2q9::VelocityView u("u", decomp.local_nx, ny, 2);

    lbm_d2q9::initialize_uniform_macroscopic_fields(rho, u, decomp.local_nx, ny, 1.0, 0.0, 0.0);
    lbm_d2q9::initialize_from_macroscopic_fields_local(rho, u, f, decomp);

    double initial_mass = 0.0;
    double initial_ke = 0.0;
    lbm_d2q9::compute_global_mass_kinetic_energy(rho, u, decomp, initial_mass, initial_ke, MPI_COMM_WORLD);

    for (int step = 0; step < 60; ++step) {
        lbm_d2q9::exchange_halo_columns(f, decomp, MPI_COMM_WORLD);
        lbm_d2q9::stream_with_cavity_boundaries_local(f, f_next, decomp, 0.05, 0.0, 1.0);
        lbm_d2q9::compute_density_local(f_next, rho, decomp);
        lbm_d2q9::compute_velocity_local(f_next, rho, u, decomp);
        lbm_d2q9::collide_bgk_local(f_next, rho, u, 1.0, decomp);
        Kokkos::fence();
        std::swap(f, f_next);
    }

    double final_mass = 0.0;
    double final_ke = 0.0;
    lbm_d2q9::compute_global_mass_kinetic_energy(rho, u, decomp, final_mass, final_ke, MPI_COMM_WORLD);

    EXPECT_NEAR(final_mass, initial_mass, 1e-10);
}

TEST(LbmMpiCavityTest, OneRankMpiMatchesSerialPath) {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 1) {
        GTEST_SKIP() << "This test is for np=1.";
    }

    constexpr int nx = 24;
    constexpr int ny = 18;
    constexpr int steps = 20;

    lbm_d2q9::DistributionView f_serial("f_serial", nx, ny, lbm_d2q9::Q);
    lbm_d2q9::DistributionView f_serial_next("f_serial_next", nx, ny, lbm_d2q9::Q);
    lbm_d2q9::DensityView rho_serial("rho_serial", nx, ny);
    lbm_d2q9::VelocityView u_serial("u_serial", nx, ny, 2);

    lbm_d2q9::initialize_uniform_macroscopic_fields(rho_serial, u_serial, nx, ny, 1.0, 0.0, 0.0);
    lbm_d2q9::initialize_from_macroscopic_fields(rho_serial, u_serial, f_serial, nx, ny);

    for (int step = 0; step < steps; ++step) {
        lbm_d2q9::stream_with_cavity_boundaries(f_serial, f_serial_next, nx, ny, 0.05, 0.0, 1.0);
        lbm_d2q9::compute_density(f_serial_next, rho_serial, nx, ny);
        lbm_d2q9::compute_velocity(f_serial_next, rho_serial, u_serial, nx, ny);
        lbm_d2q9::collide_bgk(f_serial_next, rho_serial, u_serial, 1.0, nx, ny);
        Kokkos::fence();
        std::swap(f_serial, f_serial_next);
    }

    const auto decomp = lbm_d2q9::decompose_domain_1d_x(nx, ny, MPI_COMM_WORLD);
    lbm_d2q9::DistributionView f_mpi("f_mpi", decomp.local_nx + 2, ny, lbm_d2q9::Q);
    lbm_d2q9::DistributionView f_mpi_next("f_mpi_next", decomp.local_nx + 2, ny, lbm_d2q9::Q);
    lbm_d2q9::DensityView rho_mpi("rho_mpi", decomp.local_nx, ny);
    lbm_d2q9::VelocityView u_mpi("u_mpi", decomp.local_nx, ny, 2);

    lbm_d2q9::initialize_uniform_macroscopic_fields(rho_mpi, u_mpi, decomp.local_nx, ny, 1.0, 0.0, 0.0);
    lbm_d2q9::initialize_from_macroscopic_fields_local(rho_mpi, u_mpi, f_mpi, decomp);

    for (int step = 0; step < steps; ++step) {
        lbm_d2q9::exchange_halo_columns(f_mpi, decomp, MPI_COMM_WORLD);
        lbm_d2q9::stream_with_cavity_boundaries_local(f_mpi, f_mpi_next, decomp, 0.05, 0.0, 1.0);
        lbm_d2q9::compute_density_local(f_mpi_next, rho_mpi, decomp);
        lbm_d2q9::compute_velocity_local(f_mpi_next, rho_mpi, u_mpi, decomp);
        lbm_d2q9::collide_bgk_local(f_mpi_next, rho_mpi, u_mpi, 1.0, decomp);
        Kokkos::fence();
        std::swap(f_mpi, f_mpi_next);
    }

    const auto h_rho_serial = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rho_serial);
    const auto h_u_serial = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u_serial);
    const auto h_rho_mpi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rho_mpi);
    const auto h_u_mpi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u_mpi);

    for (int x = 0; x < nx; ++x) {
        for (int y = 0; y < ny; ++y) {
            EXPECT_NEAR(h_rho_mpi(x, y), h_rho_serial(x, y), 1e-12);
            EXPECT_NEAR(h_u_mpi(x, y, 0), h_u_serial(x, y, 0), 1e-12);
            EXPECT_NEAR(h_u_mpi(x, y, 1), h_u_serial(x, y, 1), 1e-12);
        }
    }
}

}  // namespace
