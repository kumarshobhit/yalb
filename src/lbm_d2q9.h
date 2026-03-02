#ifndef LBM_D2Q9_H
#define LBM_D2Q9_H

#include <Kokkos_Core.hpp>

#include <string>

namespace lbm_d2q9 {

constexpr int Q = 9;

using DistributionView = Kokkos::View<double***>;
using DensityView = Kokkos::View<double**>;
using VelocityView = Kokkos::View<double***>;

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

void initialize_single_packet(DistributionView f, int x, int y, int direction, double value);
void streaming(const DistributionView& f_in, DistributionView& f_out, int nx, int ny);
void compute_density(const DistributionView& f, DensityView& rho, int nx, int ny);
void compute_velocity(const DistributionView& f, const DensityView& rho, VelocityView& u, int nx, int ny);

void write_scalar_csv(const std::string& filename, const DensityView& field, int nx, int ny);
void write_velocity_csv_pair(
    const std::string& ux_filename,
    const std::string& uy_filename,
    const VelocityView& u,
    int nx,
    int ny);

}  // namespace lbm_d2q9

#endif  // LBM_D2Q9_H
