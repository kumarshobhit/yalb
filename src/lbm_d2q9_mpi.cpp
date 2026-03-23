#include "lbm_d2q9_mpi.h"

#ifdef WITH_MPI

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace lbm_d2q9 {
namespace {

int local_nx_for_rank(int nx, int size, int rank) {
    const int base = nx / size;
    const int remainder = nx % size;
    return base + (rank < remainder ? 1 : 0);
}

std::vector<int> make_counts_x(int nx, int ny, int size) {
    std::vector<int> counts(size, 0);
    for (int r = 0; r < size; ++r) {
        counts[r] = local_nx_for_rank(nx, size, r) * ny;
    }
    return counts;
}

std::vector<int> make_displs_x(int nx, int ny, int size) {
    std::vector<int> displs(size, 0);
    int offset_x = 0;
    for (int r = 0; r < size; ++r) {
        displs[r] = offset_x * ny;
        offset_x += local_nx_for_rank(nx, size, r);
    }
    return displs;
}


}  // namespace

DomainDecomposition1D decompose_domain_1d_x(int nx, int ny, MPI_Comm comm) {
    DomainDecomposition1D decomp;
    decomp.global_nx = nx;
    decomp.global_ny = ny;

    MPI_Comm_rank(comm, &decomp.rank);
    MPI_Comm_size(comm, &decomp.size);

    if (decomp.size <= 0) {
        throw std::runtime_error("MPI communicator has invalid size.");
    }
    if (nx < decomp.size) {
        throw std::runtime_error("Global nx must satisfy nx >= number of MPI ranks.");
    }

    decomp.local_nx = local_nx_for_rank(nx, decomp.size, decomp.rank);

    int x_offset = 0;
    for (int r = 0; r < decomp.rank; ++r) {
        x_offset += local_nx_for_rank(nx, decomp.size, r);
    }
    decomp.x_offset = x_offset;

    decomp.left_rank = (decomp.rank == 0) ? MPI_PROC_NULL : decomp.rank - 1;
    decomp.right_rank = (decomp.rank == decomp.size - 1) ? MPI_PROC_NULL : decomp.rank + 1;

    return decomp;
}

void initialize_from_macroscopic_fields_local(
    const DensityView& rho,
    const VelocityView& u,
    DistributionView& f_with_ghost,
    const DomainDecomposition1D& decomp) {
    Kokkos::parallel_for(
        "initialize_from_macroscopic_fields_local",
        Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>({0, 0}, {decomp.local_nx, decomp.global_ny}),
        KOKKOS_LAMBDA(const int x_owned, const int y) {
            const int x_local = x_owned + 1;
            const double density = rho(x_owned, y);
            const double ux = u(x_owned, y, 0);
            const double uy = u(x_owned, y, 1);
            for (int direction = 0; direction < Q; ++direction) {
                f_with_ghost(x_local, y, direction) = equilibrium_population(density, ux, uy, direction);
            }
        });

    Kokkos::parallel_for(
        "initialize_ghost_columns",
        Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>({0, 0}, {decomp.global_ny, Q}),
        KOKKOS_LAMBDA(const int y, const int direction) {
            f_with_ghost(0, y, direction) = 0.0;
            f_with_ghost(decomp.local_nx + 1, y, direction) = 0.0;
        });
}

void exchange_halo_columns(DistributionView& f_owned_with_ghost, const DomainDecomposition1D& decomp, MPI_Comm comm) {
    const int ny = decomp.global_ny;
    const int width = ny * Q;

    if (decomp.left_rank == MPI_PROC_NULL && decomp.right_rank == MPI_PROC_NULL) {
        return;
    }

    // halo exchange for multi rank only 
    Kokkos::View<double*, MemorySpace> send_left_device("send_left_device", width);
    Kokkos::View<double*, MemorySpace> send_right_device("send_right_device", width);

    Kokkos::parallel_for(
        "pack_halo_columns",
        Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>({0, 0}, {ny, Q}),
        KOKKOS_LAMBDA(const int y, const int direction) {
            const int idx = y * Q + direction;
            send_left_device(idx) = f_owned_with_ghost(1, y, direction);
            send_right_device(idx) = f_owned_with_ghost(decomp.local_nx, y, direction);
        });

    auto send_left_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_left_device);
    auto send_right_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_right_device);

    std::vector<double> recv_left(width, 0.0);
    std::vector<double> recv_right(width, 0.0);

    MPI_Sendrecv(
        send_left_host.data(),
        width,
        MPI_DOUBLE,
        decomp.left_rank,
        101,
        recv_right.data(),
        width,
        MPI_DOUBLE,
        decomp.right_rank,
        101,
        comm,
        MPI_STATUS_IGNORE);

    MPI_Sendrecv(
        send_right_host.data(),
        width,
        MPI_DOUBLE,
        decomp.right_rank,
        202,
        recv_left.data(),
        width,
        MPI_DOUBLE,
        decomp.left_rank,
        202,
        comm,
        MPI_STATUS_IGNORE);

    Kokkos::View<double*, MemorySpace> recv_left_device("recv_left_device", width);
    Kokkos::View<double*, MemorySpace> recv_right_device("recv_right_device", width);

    if (decomp.left_rank != MPI_PROC_NULL) {
        auto recv_left_host = Kokkos::create_mirror_view(recv_left_device);
        for (int idx = 0; idx < width; ++idx) {
            recv_left_host(idx) = recv_left[idx];
        }
        Kokkos::deep_copy(recv_left_device, recv_left_host);

        Kokkos::parallel_for(
            "unpack_left_halo",
            Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>({0, 0}, {ny, Q}),
            KOKKOS_LAMBDA(const int y, const int direction) {
                const int idx = y * Q + direction;
                f_owned_with_ghost(0, y, direction) = recv_left_device(idx);
            });
    }

    if (decomp.right_rank != MPI_PROC_NULL) {
        auto recv_right_host = Kokkos::create_mirror_view(recv_right_device);
        for (int idx = 0; idx < width; ++idx) {
            recv_right_host(idx) = recv_right[idx];
        }
        Kokkos::deep_copy(recv_right_device, recv_right_host);

        Kokkos::parallel_for(
            "unpack_right_halo",
            Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>({0, 0}, {ny, Q}),
            KOKKOS_LAMBDA(const int y, const int direction) {
                const int idx = y * Q + direction;
                f_owned_with_ghost(decomp.local_nx + 1, y, direction) = recv_right_device(idx);
            });
    }
}

void stream_with_cavity_boundaries_local(
    const DistributionView& f_in,
    DistributionView& f_out,
    const DomainDecomposition1D& decomp,
    double lid_ux,
    double lid_uy,
    double rho_wall) {
    const int nx = decomp.global_nx;
    const int ny = decomp.global_ny;

    Kokkos::parallel_for(
        "stream_with_cavity_boundaries_local",
        Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>({1, 0}, {decomp.local_nx + 1, ny}),
        KOKKOS_LAMBDA(const int x_local, const int y) {
            const int x_global = decomp.x_offset + (x_local - 1);

            for (int direction = 0; direction < Q; ++direction) {
                const int source_x_global = x_global - cx(direction);
                const int source_y = y - cy(direction);

                const bool source_outside_x = (source_x_global < 0 || source_x_global >= nx);
                const bool source_outside_y = (source_y < 0 || source_y >= ny);

                if (!source_outside_x && !source_outside_y) {
                    const int source_x_local = source_x_global - decomp.x_offset + 1;
                    f_out(x_local, y, direction) = f_in(source_x_local, source_y, direction);
                    continue;
                }

                const int incoming = opposite(direction);
                double bounced = f_in(x_local, y, incoming);

                if (source_y >= ny) {
                    const double c_dot_u =
                        static_cast<double>(cx(incoming)) * lid_ux + static_cast<double>(cy(incoming)) * lid_uy;
                    bounced -= 6.0 * weight(incoming) * rho_wall * c_dot_u;
                }

                f_out(x_local, y, direction) = bounced;
            }
        });
}

void compute_density_local(
    const DistributionView& f_with_ghost,
    DensityView& rho,
    const DomainDecomposition1D& decomp) {
    Kokkos::parallel_for(
        "compute_density_local",
        Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>({0, 0}, {decomp.local_nx, decomp.global_ny}),
        KOKKOS_LAMBDA(const int x_owned, const int y) {
            const int x_local = x_owned + 1;
            double density = 0.0;
            for (int direction = 0; direction < Q; ++direction) {
                density += f_with_ghost(x_local, y, direction);
            }
            rho(x_owned, y) = density;
        });
}

void compute_velocity_local(
    const DistributionView& f_with_ghost,
    const DensityView& rho,
    VelocityView& u,
    const DomainDecomposition1D& decomp) {
    Kokkos::parallel_for(
        "compute_velocity_local",
        Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>({0, 0}, {decomp.local_nx, decomp.global_ny}),
        KOKKOS_LAMBDA(const int x_owned, const int y) {
            const int x_local = x_owned + 1;
            double momentum_x = 0.0;
            double momentum_y = 0.0;
            for (int direction = 0; direction < Q; ++direction) {
                const double value = f_with_ghost(x_local, y, direction);
                momentum_x += value * cx(direction);
                momentum_y += value * cy(direction);
            }

            const double density = rho(x_owned, y);
            if (density <= 1e-12) {
                u(x_owned, y, 0) = 0.0;
                u(x_owned, y, 1) = 0.0;
                return;
            }

            u(x_owned, y, 0) = momentum_x / density;
            u(x_owned, y, 1) = momentum_y / density;
        });
}

void collide_bgk_local(
    DistributionView& f_with_ghost,
    const DensityView& rho,
    const VelocityView& u,
    double omega,
    const DomainDecomposition1D& decomp) {
    Kokkos::parallel_for(
        "collide_bgk_local",
        Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>({0, 0}, {decomp.local_nx, decomp.global_ny}),
        KOKKOS_LAMBDA(const int x_owned, const int y) {
            const int x_local = x_owned + 1;
            const double density = rho(x_owned, y);
            const double ux = u(x_owned, y, 0);
            const double uy = u(x_owned, y, 1);

            for (int direction = 0; direction < Q; ++direction) {
                const double f_eq = equilibrium_population(density, ux, uy, direction);
                f_with_ghost(x_local, y, direction) += omega * (f_eq - f_with_ghost(x_local, y, direction));
            }
        });
}

void compute_global_mass_kinetic_energy(
    const DensityView& rho,
    const VelocityView& u,
    const DomainDecomposition1D& decomp,
    double& global_mass,
    double& global_ke,
    MPI_Comm comm) {
    const auto host_rho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rho);
    const auto host_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u);

    double local_mass = 0.0;
    double local_ke = 0.0;

    for (int x = 0; x < decomp.local_nx; ++x) {
        for (int y = 0; y < decomp.global_ny; ++y) {
            const double density = host_rho(x, y);
            const double ux = host_u(x, y, 0);
            const double uy = host_u(x, y, 1);
            local_mass += density;
            local_ke += 0.5 * density * (ux * ux + uy * uy);
        }
    }

    MPI_Allreduce(&local_mass, &global_mass, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&local_ke, &global_ke, 1, MPI_DOUBLE, MPI_SUM, comm);
}

double compute_global_max_delta_u(
    const VelocityView& previous,
    const VelocityView& current,
    const DomainDecomposition1D& decomp,
    MPI_Comm comm) {
    const auto host_prev = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), previous);
    const auto host_curr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), current);

    double local_max_delta = 0.0;
    for (int x = 0; x < decomp.local_nx; ++x) {
        for (int y = 0; y < decomp.global_ny; ++y) {
            const double dux = host_curr(x, y, 0) - host_prev(x, y, 0);
            const double duy = host_curr(x, y, 1) - host_prev(x, y, 1);
            local_max_delta = std::max(local_max_delta, std::sqrt(dux * dux + duy * duy));
        }
    }

    double global_max_delta = 0.0;
    MPI_Allreduce(&local_max_delta, &global_max_delta, 1, MPI_DOUBLE, MPI_MAX, comm);
    return global_max_delta;
}

void gather_fields_to_root(
    const DensityView& rho_local,
    const VelocityView& u_local,
    DensityView& rho_global_root,
    VelocityView& u_global_root,
    const DomainDecomposition1D& decomp,
    int root,
    MPI_Comm comm) {
    const auto host_rho_local = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rho_local);
    const auto host_u_local = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u_local);

    const int local_count = decomp.local_nx * decomp.global_ny;
    std::vector<double> rho_send(local_count, 0.0);
    std::vector<double> ux_send(local_count, 0.0);
    std::vector<double> uy_send(local_count, 0.0);

    for (int x = 0; x < decomp.local_nx; ++x) {
        for (int y = 0; y < decomp.global_ny; ++y) {
            const int idx = x * decomp.global_ny + y;
            rho_send[idx] = host_rho_local(x, y);
            ux_send[idx] = host_u_local(x, y, 0);
            uy_send[idx] = host_u_local(x, y, 1);
        }
    }

    const std::vector<int> counts = make_counts_x(decomp.global_nx, decomp.global_ny, decomp.size);
    const std::vector<int> displs = make_displs_x(decomp.global_nx, decomp.global_ny, decomp.size);

    std::vector<double> rho_recv;
    std::vector<double> ux_recv;
    std::vector<double> uy_recv;

    if (decomp.rank == root) {
        rho_recv.resize(decomp.global_nx * decomp.global_ny, 0.0);
        ux_recv.resize(decomp.global_nx * decomp.global_ny, 0.0);
        uy_recv.resize(decomp.global_nx * decomp.global_ny, 0.0);
    }

    MPI_Gatherv(
        rho_send.data(),
        local_count,
        MPI_DOUBLE,
        decomp.rank == root ? rho_recv.data() : nullptr,
        counts.data(),
        displs.data(),
        MPI_DOUBLE,
        root,
        comm);
    MPI_Gatherv(
        ux_send.data(),
        local_count,
        MPI_DOUBLE,
        decomp.rank == root ? ux_recv.data() : nullptr,
        counts.data(),
        displs.data(),
        MPI_DOUBLE,
        root,
        comm);
    MPI_Gatherv(
        uy_send.data(),
        local_count,
        MPI_DOUBLE,
        decomp.rank == root ? uy_recv.data() : nullptr,
        counts.data(),
        displs.data(),
        MPI_DOUBLE,
        root,
        comm);

    if (decomp.rank != root) {
        return;
    }

    auto host_rho_global = Kokkos::create_mirror_view(rho_global_root);
    auto host_u_global = Kokkos::create_mirror_view(u_global_root);

    for (int x = 0; x < decomp.global_nx; ++x) {
        for (int y = 0; y < decomp.global_ny; ++y) {
            const int idx = x * decomp.global_ny + y;
            host_rho_global(x, y) = rho_recv[idx];
            host_u_global(x, y, 0) = ux_recv[idx];
            host_u_global(x, y, 1) = uy_recv[idx];
        }
    }

    Kokkos::deep_copy(rho_global_root, host_rho_global);
    Kokkos::deep_copy(u_global_root, host_u_global);
}

}  // namespace lbm_d2q9

#endif  // WITH_MPI
