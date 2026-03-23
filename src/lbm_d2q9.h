#ifndef LBM_D2Q9_H
#define LBM_D2Q9_H

#include <Kokkos_Core.hpp>

#include <string>

namespace lbm_d2q9 {

constexpr int Q = 9;

#if defined(KOKKOS_ENABLE_CUDA)
using ExecutionSpace = Kokkos::Cuda;
#else
using ExecutionSpace = Kokkos::DefaultExecutionSpace;
#endif
using MemorySpace = ExecutionSpace::memory_space;

using DistributionView = Kokkos::View<double***, MemorySpace>;
using DensityView = Kokkos::View<double**, MemorySpace>;
using VelocityView = Kokkos::View<double***, MemorySpace>;

constexpr double pi = 3.14159265358979323846;

KOKKOS_INLINE_FUNCTION
double weight(int direction) {
    constexpr double values[Q] = {4.0 / 9.0,
                                  1.0 / 9.0,
                                  1.0 / 9.0,
                                  1.0 / 9.0,
                                  1.0 / 9.0,
                                  1.0 / 36.0,
                                  1.0 / 36.0,
                                  1.0 / 36.0,
                                  1.0 / 36.0};
    return values[direction];
}

KOKKOS_INLINE_FUNCTION
int cx(int direction) {
    constexpr int values[Q] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
    return values[direction];
}

KOKKOS_INLINE_FUNCTION
int cy(int direction) {
    constexpr int values[Q] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
    return values[direction];
}

KOKKOS_INLINE_FUNCTION
int opposite(int direction) {
    constexpr int values[Q] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
    return values[direction];
}

KOKKOS_INLINE_FUNCTION
double equilibrium_population(double rho, double ux, double uy, int direction) {
    const double c_dot_u = static_cast<double>(cx(direction)) * ux + static_cast<double>(cy(direction)) * uy;
    const double u_squared = ux * ux + uy * uy;
    return weight(direction) * rho * (1.0 + 3.0 * c_dot_u + 4.5 * c_dot_u * c_dot_u - 1.5 * u_squared);
}

void initialize_single_packet(DistributionView f, int x, int y, int direction, double value);
void initialize_uniform_macroscopic_fields(
    DensityView rho,
    VelocityView u,
    int nx,
    int ny,
    double rho0,
    double ux0,
    double uy0);
void initialize_shear_wave_macroscopic_fields(
    DensityView rho,
    VelocityView u,
    int nx,
    int ny,
    double rho0,
    double amplitude);
void initialize_from_macroscopic_fields(const DensityView& rho, const VelocityView& u, DistributionView& f, int nx, int ny);
void streaming(const DistributionView& f_in, DistributionView& f_out, int nx, int ny);
void stream_with_cavity_boundaries(
    const DistributionView& f_in,
    DistributionView& f_out,
    int nx,
    int ny,
    double lid_ux,
    double lid_uy,
    double rho_wall);
void compute_density(const DistributionView& f, DensityView& rho, int nx, int ny);
void compute_velocity(const DistributionView& f, const DensityView& rho, VelocityView& u, int nx, int ny);
void collide_bgk(DistributionView& f, const DensityView& rho, const VelocityView& u, double omega, int nx, int ny);
double theoretical_viscosity_from_omega(double omega);
double analytical_shear_wave_amplitude(double initial_amplitude, double viscosity, int ny, int step);
double measure_shear_wave_amplitude(const VelocityView& u, int nx, int ny);

void write_scalar_csv(const std::string& filename, const DensityView& field, int nx, int ny);
void write_velocity_csv_pair(
    const std::string& ux_filename,
    const std::string& uy_filename,
    const VelocityView& u,
    int nx,
    int ny);

}  // namespace lbm_d2q9

#endif  // LBM_D2Q9_H
