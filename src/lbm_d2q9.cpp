#include "lbm_d2q9.h"

#include <fstream>

namespace lbm_d2q9 {

void initialize_single_packet(DistributionView f, int x, int y, int direction, double value) {
    Kokkos::deep_copy(f, 0.0);

    auto host_f = Kokkos::create_mirror_view(f);
    host_f(x, y, direction) = value;
    Kokkos::deep_copy(f, host_f);
}

void streaming(const DistributionView& f_in, DistributionView& f_out, int nx, int ny) {
    Kokkos::parallel_for(
        "streaming",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nx, ny}),
        KOKKOS_LAMBDA(const int x, const int y) {
            for (int direction = 0; direction < Q; ++direction) {
                int neighbor_x = x - cx(direction);
                int neighbor_y = y - cy(direction);

                if (neighbor_x < 0) {
                    neighbor_x += nx;
                } else if (neighbor_x >= nx) {
                    neighbor_x -= nx;
                }

                if (neighbor_y < 0) {
                    neighbor_y += ny;
                } else if (neighbor_y >= ny) {
                    neighbor_y -= ny;
                }

                f_out(x, y, direction) = f_in(neighbor_x, neighbor_y, direction);
            }
        });
}

void compute_density(const DistributionView& f, DensityView& rho, int nx, int ny) {
    Kokkos::parallel_for(
        "compute_density",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nx, ny}),
        KOKKOS_LAMBDA(const int x, const int y) {
            double density = 0.0;
            for (int direction = 0; direction < Q; ++direction) {
                density += f(x, y, direction);
            }
            rho(x, y) = density;
        });
}

void compute_velocity(const DistributionView& f, const DensityView& rho, VelocityView& u, int nx, int ny) {
    Kokkos::parallel_for(
        "compute_velocity",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nx, ny}),
        KOKKOS_LAMBDA(const int x, const int y) {
            double momentum_x = 0.0;
            double momentum_y = 0.0;

            for (int direction = 0; direction < Q; ++direction) {
                const double value = f(x, y, direction);
                momentum_x += value * cx(direction);
                momentum_y += value * cy(direction);
            }

            const double density = rho(x, y);
            if (density <= 1e-12) {
                u(x, y, 0) = 0.0;
                u(x, y, 1) = 0.0;
                return;
            }

            u(x, y, 0) = momentum_x / density;
            u(x, y, 1) = momentum_y / density;
        });
}

void write_scalar_csv(const std::string& filename, const DensityView& field, int nx, int ny) {
    const auto host_field = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), field);

    std::ofstream file(filename);
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            file << host_field(x, y);
            if (x + 1 < nx) {
                file << ",";
            }
        }
        file << "\n";
    }
}

void write_velocity_csv_pair(
    const std::string& ux_filename,
    const std::string& uy_filename,
    const VelocityView& u,
    int nx,
    int ny) {
    const auto host_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u);

    std::ofstream ux_file(ux_filename);
    std::ofstream uy_file(uy_filename);

    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            ux_file << host_u(x, y, 0);
            uy_file << host_u(x, y, 1);

            if (x + 1 < nx) {
                ux_file << ",";
                uy_file << ",";
            }
        }
        ux_file << "\n";
        uy_file << "\n";
    }
}

}  // namespace lbm_d2q9
