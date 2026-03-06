#include "lbm_d2q9.h"

#include <Kokkos_Core.hpp>

#include <algorithm>
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
    std::filesystem::path output_dir = "outputs_m5";
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
              << "  --residual-every <n>     Report max-delta-u every n steps (default: 1000)\n"
              << "  --output-dir <path>      Directory for outputs (default: outputs_m5)\n"
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

double compute_max_delta_velocity(
    const lbm_d2q9::VelocityView& previous,
    const lbm_d2q9::VelocityView& current,
    int nx,
    int ny) {
    const auto host_prev = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), previous);
    const auto host_curr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), current);

    double max_delta = 0.0;
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            const double dux = host_curr(x, y, 0) - host_prev(x, y, 0);
            const double duy = host_curr(x, y, 1) - host_prev(x, y, 1);
            max_delta = std::max(max_delta, std::sqrt(dux * dux + duy * duy));
        }
    }
    return max_delta;
}

void write_fields(const SimulationConfig& config, int step, const lbm_d2q9::DensityView& rho, const lbm_d2q9::VelocityView& u) {
    lbm_d2q9::write_scalar_csv(make_output_name(config.output_dir, "rho", step), rho, config.nx, config.ny);
    lbm_d2q9::write_velocity_csv_pair(
        make_output_name(config.output_dir, "ux", step),
        make_output_name(config.output_dir, "uy", step),
        u,
        config.nx,
        config.ny);
}

}  // namespace

int main(int argc, char* argv[]) {
    const SimulationConfig config = [&]() {
        try {
            return parse_arguments(argc, argv);
        } catch (const std::exception& error) {
            std::cerr << error.what() << std::endl;
            print_usage(argv[0]);
            std::exit(1);
        }
    }();

    Kokkos::initialize(argc, argv);
    {
        std::cout << "Initializing Milestone 05 (lid-driven cavity)" << std::endl;
        std::cout << "Grid: " << config.nx << "x" << config.ny << ", steps: " << config.steps
                  << ", omega: " << config.omega << ", rho0: " << config.rho0 << std::endl;
        std::cout << "Lid velocity: (" << config.lid_ux << ", " << config.lid_uy << ")"
                  << ", write every " << config.write_field_every
                  << ", residual every " << config.residual_every << std::endl;

        lbm_d2q9::DistributionView f("f", config.nx, config.ny, lbm_d2q9::Q);
        lbm_d2q9::DistributionView f_next("f_next", config.nx, config.ny, lbm_d2q9::Q);
        lbm_d2q9::DensityView rho("rho", config.nx, config.ny);
        lbm_d2q9::VelocityView u("u", config.nx, config.ny, 2);
        lbm_d2q9::VelocityView u_prev_sample("u_prev_sample", config.nx, config.ny, 2);

        std::filesystem::create_directories(config.output_dir);
        std::ofstream residual_file(config.output_dir / "residual_history.csv");
        residual_file << "step,max_delta_u\n";

        lbm_d2q9::initialize_uniform_macroscopic_fields(rho, u, config.nx, config.ny, config.rho0, 0.0, 0.0);
        lbm_d2q9::initialize_from_macroscopic_fields(rho, u, f, config.nx, config.ny);

        Kokkos::deep_copy(u_prev_sample, u);
        residual_file << 0 << "," << 0.0 << "\n";

        write_fields(config, 0, rho, u);

        for (int step = 1; step <= config.steps; ++step) {
            lbm_d2q9::stream_with_cavity_boundaries(
                f,
                f_next,
                config.nx,
                config.ny,
                config.lid_ux,
                config.lid_uy,
                config.rho0);
            lbm_d2q9::compute_density(f_next, rho, config.nx, config.ny);
            lbm_d2q9::compute_velocity(f_next, rho, u, config.nx, config.ny);
            lbm_d2q9::collide_bgk(f_next, rho, u, config.omega, config.nx, config.ny);
            Kokkos::fence();
            std::swap(f, f_next);

            if (step % config.residual_every == 0) {
                const double max_delta_u = compute_max_delta_velocity(u_prev_sample, u, config.nx, config.ny);
                residual_file << step << "," << max_delta_u << "\n";
                Kokkos::deep_copy(u_prev_sample, u);
                std::cout << "Step " << step << ": max_delta_u = " << max_delta_u << std::endl;
            }

            if (step % config.write_field_every == 0) {
                write_fields(config, step, rho, u);
            }
        }

        if (config.steps % config.write_field_every != 0) {
            write_fields(config, config.steps, rho, u);
        }
    }
    Kokkos::finalize();
    return 0;
}
