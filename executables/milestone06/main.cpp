#include "lbm_d2q9.h"
#include "lbm_d2q9_mpi.h"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

struct SimulationConfig {
    int nx = 128;
    int ny = 128;
    int steps = 20000;
    double omega = 1.0;
    double rho0 = 1.0;
    double lid_ux = 0.05;
    double lid_uy = 0.0;
    int write_field_every = 1000;
    int residual_every = 1000;
    std::filesystem::path output_dir = "outputs_m6";
};

std::string make_output_name(const std::filesystem::path& directory, const std::string& prefix, int step) {
    std::ostringstream filename;
    filename << prefix << "_step_" << std::setw(5) << std::setfill('0') << step << ".csv";
    return (directory / filename.str()).string();
}

int parse_int_argument(const std::string& option, const char* value) {
    try {
        return std::stoi(value);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid integer value for " + option + ": " + value);
    }
}

double parse_double_argument(const std::string& option, const char* value) {
    try {
        return std::stod(value);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid floating-point value for " + option + ": " + value);
    }
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "  --nx <int>               Grid width (default: 128)\n"
              << "  --ny <int>               Grid height (default: 128)\n"
              << "  --steps <int>            Number of time steps (default: 20000)\n"
              << "  --omega <double>         BGK relaxation parameter, 0 < omega < 2 (default: 1.0)\n"
              << "  --rho0 <double>          Initial density and wall density (default: 1.0)\n"
              << "  --lid-ux <double>        Top-lid x velocity, |u| < 0.1 (default: 0.05)\n"
              << "  --lid-uy <double>        Top-lid y velocity, |u| < 0.1 (default: 0.0)\n"
              << "  --write-field-every <n>  Write rho/ux/uy every n steps (default: 1000)\n"
              << "  --residual-every <n>     Report residual every n steps (default: 1000)\n"
              << "  --output-dir <path>      Directory for outputs (default: outputs_m6)\n"
              << "  --help                   Show this help message\n";
}

SimulationConfig parse_arguments(int argc, char* argv[]) {
    SimulationConfig config;

    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        auto require_value = [&](const std::string& current_option) -> const char* {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + current_option);
            }
            return argv[++i];
        };

        if (option == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (option == "--nx") {
            config.nx = parse_int_argument(option, require_value(option));
        } else if (option == "--ny") {
            config.ny = parse_int_argument(option, require_value(option));
        } else if (option == "--steps") {
            config.steps = parse_int_argument(option, require_value(option));
        } else if (option == "--omega") {
            config.omega = parse_double_argument(option, require_value(option));
        } else if (option == "--rho0") {
            config.rho0 = parse_double_argument(option, require_value(option));
        } else if (option == "--lid-ux") {
            config.lid_ux = parse_double_argument(option, require_value(option));
        } else if (option == "--lid-uy") {
            config.lid_uy = parse_double_argument(option, require_value(option));
        } else if (option == "--write-field-every") {
            config.write_field_every = parse_int_argument(option, require_value(option));
        } else if (option == "--residual-every") {
            config.residual_every = parse_int_argument(option, require_value(option));
        } else if (option == "--output-dir") {
            config.output_dir = require_value(option);
        } else {
            throw std::runtime_error("Unknown option: " + option);
        }
    }

    if (config.nx <= 2 || config.ny <= 2) {
        throw std::runtime_error("Grid dimensions must satisfy nx > 2 and ny > 2.");
    }
    if (config.steps < 0) {
        throw std::runtime_error("Number of steps must be non-negative.");
    }
    if (!(config.omega > 0.0 && config.omega < 2.0)) {
        throw std::runtime_error("Omega must satisfy 0 < omega < 2.");
    }
    if (!(config.rho0 > 0.0)) {
        throw std::runtime_error("rho0 must be positive.");
    }
    if (std::abs(config.lid_ux) >= 0.1 || std::abs(config.lid_uy) >= 0.1) {
        throw std::runtime_error("lid-ux and lid-uy must satisfy |u| < 0.1.");
    }
    if (config.write_field_every <= 0) {
        throw std::runtime_error("write-field-every must be positive.");
    }
    if (config.residual_every <= 0) {
        throw std::runtime_error("residual-every must be positive.");
    }

    return config;
}

void write_global_fields(
    int step,
    const SimulationConfig& config,
    const lbm_d2q9::DensityView& rho_global,
    const lbm_d2q9::VelocityView& u_global) {
    lbm_d2q9::write_scalar_csv(make_output_name(config.output_dir, "rho", step), rho_global, config.nx, config.ny);
    lbm_d2q9::write_velocity_csv_pair(
        make_output_name(config.output_dir, "ux", step),
        make_output_name(config.output_dir, "uy", step),
        u_global,
        config.nx,
        config.ny);
}

}  // namespace

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const SimulationConfig config = [&]() {
        try {
            return parse_arguments(argc, argv);
        } catch (const std::exception& error) {
            if (rank == 0) {
                std::cerr << error.what() << std::endl;
                print_usage(argv[0]);
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
            std::exit(1);
        }
    }();

    if (size > config.nx) {
        if (rank == 0) {
            std::cerr << "Number of MPI ranks must satisfy size <= nx." << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::exit(1);
    }

    Kokkos::initialize(argc, argv);
    {
        if (rank == 0) {
            std::cout << "Kokkos DefaultExecutionSpace: " << Kokkos::DefaultExecutionSpace::name() << std::endl;
            std::cout << "LBM Selected ExecutionSpace: " << lbm_d2q9::ExecutionSpace::name() << std::endl;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        for (int r = 0; r < size; ++r) {
            if (rank == r) {
                const char* cuda_visible_devices = std::getenv("CUDA_VISIBLE_DEVICES");
                std::cout << "Rank " << rank
                          << " CUDA_VISIBLE_DEVICES=" << (cuda_visible_devices ? cuda_visible_devices : "<unset>")
                          << std::endl;
                Kokkos::print_configuration(std::cout, false);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        const auto decomp = lbm_d2q9::decompose_domain_1d_x(config.nx, config.ny, MPI_COMM_WORLD);

        if (rank == 0) {
            std::cout << "Initializing Milestone 06 (MPI cavity)" << std::endl;
            std::cout << "MPI ranks: " << size << ", global grid: " << config.nx << "x" << config.ny
                      << ", steps: " << config.steps << std::endl;
            std::cout << "omega: " << config.omega << ", rho0: " << config.rho0
                      << ", lid velocity: (" << config.lid_ux << ", " << config.lid_uy << ")" << std::endl;
        }

        lbm_d2q9::DistributionView f("f", decomp.local_nx + 2, config.ny, lbm_d2q9::Q);
        lbm_d2q9::DistributionView f_next("f_next", decomp.local_nx + 2, config.ny, lbm_d2q9::Q);
        lbm_d2q9::DensityView rho("rho", decomp.local_nx, config.ny);
        lbm_d2q9::VelocityView u("u", decomp.local_nx, config.ny, 2);
        lbm_d2q9::VelocityView u_prev_sample("u_prev_sample", decomp.local_nx, config.ny, 2);

        lbm_d2q9::DensityView rho_global;
        lbm_d2q9::VelocityView u_global;
        if (rank == 0) {
            rho_global = lbm_d2q9::DensityView("rho_global", config.nx, config.ny);
            u_global = lbm_d2q9::VelocityView("u_global", config.nx, config.ny, 2);
            std::filesystem::create_directories(config.output_dir);
        }

        std::ofstream residual_file;
        if (rank == 0) {
            residual_file.open(config.output_dir / "residual_history.csv");
            residual_file << "step,max_delta_u,global_mass,global_ke\n";
        }

        lbm_d2q9::initialize_uniform_macroscopic_fields(rho, u, decomp.local_nx, config.ny, config.rho0, 0.0, 0.0);
        lbm_d2q9::initialize_from_macroscopic_fields_local(rho, u, f, decomp);
        Kokkos::deep_copy(u_prev_sample, u);

        double global_mass = 0.0;
        double global_ke = 0.0;
        lbm_d2q9::compute_global_mass_kinetic_energy(rho, u, decomp, global_mass, global_ke, MPI_COMM_WORLD);

        if (rank == 0) {
            residual_file << 0 << "," << 0.0 << "," << global_mass << "," << global_ke << "\n";
        }

        lbm_d2q9::gather_fields_to_root(rho, u, rho_global, u_global, decomp, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            write_global_fields(0, config, rho_global, u_global);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        const double t_start = MPI_Wtime();

        for (int step = 1; step <= config.steps; ++step) {
            lbm_d2q9::exchange_halo_columns(f, decomp, MPI_COMM_WORLD);
            lbm_d2q9::stream_with_cavity_boundaries_local(
                f,
                f_next,
                decomp,
                config.lid_ux,
                config.lid_uy,
                config.rho0);
            lbm_d2q9::compute_density_local(f_next, rho, decomp);
            lbm_d2q9::compute_velocity_local(f_next, rho, u, decomp);
            lbm_d2q9::collide_bgk_local(f_next, rho, u, config.omega, decomp);
            Kokkos::fence();
            std::swap(f, f_next);

            if (step % config.residual_every == 0) {
                const double max_delta_u = lbm_d2q9::compute_global_max_delta_u(u_prev_sample, u, decomp, MPI_COMM_WORLD);
                lbm_d2q9::compute_global_mass_kinetic_energy(rho, u, decomp, global_mass, global_ke, MPI_COMM_WORLD);
                Kokkos::deep_copy(u_prev_sample, u);

                if (rank == 0) {
                    residual_file << step << "," << max_delta_u << "," << global_mass << "," << global_ke << "\n";
                    std::cout << "Step " << step << ": max_delta_u = " << max_delta_u
                              << ", global_mass = " << global_mass << ", global_ke = " << global_ke << std::endl;
                }
            }

            if (step % config.write_field_every == 0) {
                lbm_d2q9::gather_fields_to_root(rho, u, rho_global, u_global, decomp, 0, MPI_COMM_WORLD);
                if (rank == 0) {
                    write_global_fields(step, config, rho_global, u_global);
                }
            }
        }

        if (config.steps % config.write_field_every != 0) {
            lbm_d2q9::gather_fields_to_root(rho, u, rho_global, u_global, decomp, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                write_global_fields(config.steps, config, rho_global, u_global);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        const double local_elapsed = MPI_Wtime() - t_start;
        double max_elapsed = 0.0;
        MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            std::cout << "Total runtime (s): " << max_elapsed << std::endl;
        }
    }
    Kokkos::finalize();

    MPI_Finalize();
    return 0;
}
