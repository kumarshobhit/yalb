#include "lbm_d2q9.h"

#include <Kokkos_Core.hpp>

#include <filesystem>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <utility>

namespace {

struct SimulationConfig {
    int nx = 41;
    int ny = 41;
    int steps = 100;
    double omega = 1.0;
    double base_rho = 0.2;
    double bump_rho = 0.25;
    double drift_ux = 0.05;
    std::string scenario = "bump";
    std::filesystem::path output_dir = "outputs_m3";
};

std::string make_output_name(const std::filesystem::path& directory, const std::string& prefix, int step) {
    std::ostringstream filename;
    filename << prefix << "_step_" << std::setw(3) << std::setfill('0') << step << ".csv";
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
              << "  --nx <int>             Grid width (default: 41)\n"
              << "  --ny <int>             Grid height (default: 41)\n"
              << "  --steps <int>          Number of time steps (default: 100)\n"
              << "  --omega <double>       BGK relaxation parameter, 0 < omega < 2 (default: 1.0)\n"
              << "  --scenario <name>      bump or drift (default: bump)\n"
              << "  --base-rho <double>    Background density (default: 0.2)\n"
              << "  --bump-rho <double>    Center density for bump scenario (default: 0.25)\n"
              << "  --drift-ux <double>    X velocity for drift scenario (default: 0.05)\n"
              << "  --output-dir <path>    Directory for CSV outputs (default: outputs_m3)\n"
              << "  --help                 Show this help message\n";
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
        } else if (option == "--scenario") {
            config.scenario = require_value(option);
        } else if (option == "--base-rho") {
            config.base_rho = parse_double_argument(option, require_value(option));
        } else if (option == "--bump-rho") {
            config.bump_rho = parse_double_argument(option, require_value(option));
        } else if (option == "--drift-ux") {
            config.drift_ux = parse_double_argument(option, require_value(option));
        } else if (option == "--output-dir") {
            config.output_dir = require_value(option);
        } else {
            throw std::runtime_error("Unknown option: " + option);
        }
    }

    if (config.nx <= 0 || config.ny <= 0) {
        throw std::runtime_error("Grid dimensions must be positive.");
    }
    if (config.steps < 0) {
        throw std::runtime_error("Number of steps must be non-negative.");
    }
    if (!(config.omega > 0.0 && config.omega < 2.0)) {
        throw std::runtime_error("Omega must satisfy 0 < omega < 2.");
    }
    if (!(config.base_rho > 0.0 && config.base_rho < 1.0)) {
        throw std::runtime_error("base-rho must satisfy 0 < rho < 1.");
    }
    if (!(config.bump_rho > 0.0 && config.bump_rho < 1.0)) {
        throw std::runtime_error("bump-rho must satisfy 0 < rho < 1.");
    }
    if (std::abs(config.drift_ux) >= 0.1) {
        throw std::runtime_error("drift-ux must satisfy |u| < 0.1.");
    }
    if (config.scenario != "bump" && config.scenario != "drift") {
        throw std::runtime_error("scenario must be either 'bump' or 'drift'.");
    }

    return config;
}

void initialize_bump_scenario(lbm_d2q9::DensityView rho, lbm_d2q9::VelocityView u, const SimulationConfig& config) {
    const int center_x = config.nx / 2;
    const int center_y = config.ny / 2;

    Kokkos::parallel_for(
        "initialize_bump",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {config.nx, config.ny}),
        KOKKOS_LAMBDA(const int x, const int y) {
            rho(x, y) = config.base_rho;
            if (x == center_x && y == center_y) {
                rho(x, y) = config.bump_rho;
            }
            u(x, y, 0) = 0.0;
            u(x, y, 1) = 0.0;
        });
}

void initialize_drift_scenario(lbm_d2q9::DensityView rho, lbm_d2q9::VelocityView u, const SimulationConfig& config) {
    const int center_x = config.nx / 2;
    const int center_y = config.ny / 2;
    const int radius_squared = 16;

    Kokkos::parallel_for(
        "initialize_drift",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {config.nx, config.ny}),
        KOKKOS_LAMBDA(const int x, const int y) {
            const int dx = x - center_x;
            const int dy = y - center_y;
            rho(x, y) = config.base_rho;
            u(x, y, 0) = 0.0;
            u(x, y, 1) = 0.0;

            if (dx * dx + dy * dy <= radius_squared) {
                rho(x, y) = config.bump_rho;
                u(x, y, 0) = config.drift_ux;
            }
        });
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
        std::cout << "Initializing Milestone 03 Simulation..." << std::endl;
        std::cout << "Grid: " << config.nx << "x" << config.ny << ", steps: " << config.steps
                  << ", omega: " << config.omega << ", scenario: " << config.scenario << std::endl;

        lbm_d2q9::DistributionView f("f", config.nx, config.ny, lbm_d2q9::Q);
        lbm_d2q9::DistributionView f_next("f_next", config.nx, config.ny, lbm_d2q9::Q);
        lbm_d2q9::DensityView rho("rho", config.nx, config.ny);
        lbm_d2q9::VelocityView u("u", config.nx, config.ny, 2);

        std::filesystem::create_directories(config.output_dir);

        if (config.scenario == "bump") {
            initialize_bump_scenario(rho, u, config);
        } else {
            initialize_drift_scenario(rho, u, config);
        }
        lbm_d2q9::initialize_from_macroscopic_fields(rho, u, f, config.nx, config.ny);

        for (int step = 0; step < config.steps; ++step) {
            lbm_d2q9::streaming(f, f_next, config.nx, config.ny);
            Kokkos::fence();

            lbm_d2q9::compute_density(f_next, rho, config.nx, config.ny);
            lbm_d2q9::compute_velocity(f_next, rho, u, config.nx, config.ny);
            lbm_d2q9::collide_bgk(f_next, rho, u, config.omega, config.nx, config.ny);
            Kokkos::fence();

            std::swap(f, f_next);

            const std::string rho_filename = make_output_name(config.output_dir, "rho", step);
            const std::string ux_filename = make_output_name(config.output_dir, "ux", step);
            const std::string uy_filename = make_output_name(config.output_dir, "uy", step);

            lbm_d2q9::write_scalar_csv(rho_filename, rho, config.nx, config.ny);
            lbm_d2q9::write_velocity_csv_pair(ux_filename, uy_filename, u, config.nx, config.ny);

            std::cout << "Step " << step << ": wrote " << rho_filename << ", " << ux_filename << ", " << uy_filename
                      << std::endl;
        }
    }
    Kokkos::finalize();
    return 0;
}
