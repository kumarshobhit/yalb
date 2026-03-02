#include "lbm_d2q9.h"

#include <Kokkos_Core.hpp>

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <utility>

namespace {

struct SimulationConfig {
    int nx = 15;
    int ny = 10;
    int steps = 10;
    int initial_x = 5;
    int initial_y = 5;
    int direction = 1;
    double value = 1.0;
    std::filesystem::path output_dir = "outputs";
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
              << "  --nx <int>            Grid width (default: 15)\n"
              << "  --ny <int>            Grid height (default: 10)\n"
              << "  --steps <int>         Number of time steps (default: 10)\n"
              << "  --x <int>             Initial packet x coordinate (default: 5)\n"
              << "  --y <int>             Initial packet y coordinate (default: 5)\n"
              << "  --direction <0-8>     Initial D2Q9 direction index (default: 1)\n"
              << "  --value <double>      Initial packet value (default: 1.0)\n"
              << "  --output-dir <path>   Directory for CSV outputs (default: outputs)\n"
              << "  --help                Show this help message\n";
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
        }
        if (option == "--nx") {
            config.nx = parse_int_argument(option, require_value(option));
        } else if (option == "--ny") {
            config.ny = parse_int_argument(option, require_value(option));
        } else if (option == "--steps") {
            config.steps = parse_int_argument(option, require_value(option));
        } else if (option == "--x") {
            config.initial_x = parse_int_argument(option, require_value(option));
        } else if (option == "--y") {
            config.initial_y = parse_int_argument(option, require_value(option));
        } else if (option == "--direction") {
            config.direction = parse_int_argument(option, require_value(option));
        } else if (option == "--value") {
            config.value = parse_double_argument(option, require_value(option));
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
    if (config.direction < 0 || config.direction >= lbm_d2q9::Q) {
        throw std::runtime_error("Direction must be between 0 and 8.");
    }
    if (config.initial_x < 0 || config.initial_x >= config.nx || config.initial_y < 0 || config.initial_y >= config.ny) {
        throw std::runtime_error("Initial packet coordinates must lie inside the grid.");
    }
    if (config.value < 0.0) {
        throw std::runtime_error("Initial packet value must be non-negative.");
    }

    return config;
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
        std::cout << "Initializing Milestone 2 Simulation..." << std::endl;
        std::cout << "Grid: " << config.nx << "x" << config.ny << ", steps: " << config.steps
                  << ", initial packet: (" << config.initial_x << ", " << config.initial_y << "), direction "
                  << config.direction << ", value " << config.value << std::endl;

        lbm_d2q9::DistributionView f("f", config.nx, config.ny, lbm_d2q9::Q);
        lbm_d2q9::DistributionView f_next("f_next", config.nx, config.ny, lbm_d2q9::Q);
        lbm_d2q9::DensityView rho("rho", config.nx, config.ny);
        lbm_d2q9::VelocityView u("u", config.nx, config.ny, 2);

        std::filesystem::create_directories(config.output_dir);

        lbm_d2q9::initialize_single_packet(f, config.initial_x, config.initial_y, config.direction, config.value);

        for (int step = 0; step < config.steps; ++step) {
            lbm_d2q9::streaming(f, f_next, config.nx, config.ny);
            Kokkos::fence();
            std::swap(f, f_next);

            lbm_d2q9::compute_density(f, rho, config.nx, config.ny);
            lbm_d2q9::compute_velocity(f, rho, u, config.nx, config.ny);
            Kokkos::fence();

            const std::string rho_filename = make_output_name(config.output_dir, "rho", step);
            const std::string ux_filename = make_output_name(config.output_dir, "ux", step);
            const std::string uy_filename = make_output_name(config.output_dir, "uy", step);

            lbm_d2q9::write_scalar_csv(rho_filename, rho, config.nx, config.ny);
            lbm_d2q9::write_velocity_csv_pair(ux_filename, uy_filename, u, config.nx, config.ny);

            std::cout << "Step " << step << ": wrote " << rho_filename << ", " << ux_filename << ", " << uy_filename
                      << std::endl;

            const auto rho_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rho);
            bool found = false;
            for (int x = 0; x < config.nx; ++x) {
                for (int y = 0; y < config.ny; ++y) {
                    if (rho_host(x, y) > 0.5) {
                        std::cout << "Step " << step << ": Particle found at (" << x << ", " << y << ")" << std::endl;
                        found = true;
                    }
                }
            }
            if (!found) {
                std::cout << "Step " << step << ": Particle lost (Mass Error!)" << std::endl;
            }
        }
    }
    Kokkos::finalize();
    return 0;
}
