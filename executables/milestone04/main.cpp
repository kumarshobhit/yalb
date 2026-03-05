#include "lbm_d2q9.h"

#include <Kokkos_Core.hpp>

#include <algorithm>
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
    int nx = 64;
    int ny = 64;
    int steps = 200;
    double omega = 1.0;
    double rho0 = 0.2;
    double amplitude = 0.05;
    int write_field_every = 10;
    std::filesystem::path output_dir = "outputs_m4";
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
              << "  --nx <int>               Grid width (default: 64)\n"
              << "  --ny <int>               Grid height (default: 64)\n"
              << "  --steps <int>            Number of time steps (default: 200)\n"
              << "  --omega <double>         BGK relaxation parameter, 0 < omega < 2 (default: 1.0)\n"
              << "  --rho0 <double>          Uniform density, 0 < rho0 < 1 (default: 0.2)\n"
              << "  --amplitude <double>     Initial shear-wave amplitude, 0 <= a0 < 0.1 (default: 0.05)\n"
              << "  --output-dir <path>      Directory for CSV outputs (default: outputs_m4)\n"
              << "  --write-field-every <n>  Write rho/ux/uy fields every n steps (default: 10)\n"
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
        } else if (option == "--amplitude") {
            config.amplitude = parse_double_argument(option, require_value(option));
        } else if (option == "--output-dir") {
            config.output_dir = require_value(option);
        } else if (option == "--write-field-every") {
            config.write_field_every = parse_int_argument(option, require_value(option));
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
    if (!(config.rho0 > 0.0 && config.rho0 < 1.0)) {
        throw std::runtime_error("rho0 must satisfy 0 < rho0 < 1.");
    }
    if (!(config.amplitude >= 0.0 && config.amplitude < 0.1)) {
        throw std::runtime_error("Amplitude must satisfy 0 <= amplitude < 0.1.");
    }
    if (config.write_field_every <= 0) {
        throw std::runtime_error("write-field-every must be positive.");
    }

    return config;
}

void write_decay_row(
    std::ofstream& file,
    int step,
    double time,
    double a_sim,
    double a_theory) {
    const double relative_error = std::abs(a_sim - a_theory) / std::max(std::abs(a_theory), 1e-14);
    file << step << "," << time << "," << a_sim << "," << a_theory << "," << relative_error << "\n";
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
        const double tau = 1.0 / config.omega;
        const double nu_theory = lbm_d2q9::theoretical_viscosity_from_omega(config.omega);

        std::cout << "Initializing Milestone 4 Simulation..." << std::endl;
        std::cout << "Grid: " << config.nx << "x" << config.ny << ", steps: " << config.steps
                  << ", amplitude: " << config.amplitude << ", rho0: " << config.rho0 << std::endl;
        std::cout << "omega = " << config.omega << ", tau = " << tau << ", nu_theory = " << nu_theory << std::endl;

        lbm_d2q9::DistributionView f("f", config.nx, config.ny, lbm_d2q9::Q);
        lbm_d2q9::DistributionView f_next("f_next", config.nx, config.ny, lbm_d2q9::Q);
        lbm_d2q9::DensityView rho("rho", config.nx, config.ny);
        lbm_d2q9::VelocityView u("u", config.nx, config.ny, 2);

        std::filesystem::create_directories(config.output_dir);
        std::ofstream decay_file(config.output_dir / "shear_decay.csv");
        decay_file << "step,time,a_sim,a_theory,relative_error\n";

        lbm_d2q9::initialize_shear_wave_macroscopic_fields(rho, u, config.nx, config.ny, config.rho0, config.amplitude);
        lbm_d2q9::initialize_from_macroscopic_fields(rho, u, f, config.nx, config.ny);

        const double initial_sim = lbm_d2q9::measure_shear_wave_amplitude(u, config.nx, config.ny);
        const double initial_theory =
            lbm_d2q9::analytical_shear_wave_amplitude(config.amplitude, nu_theory, config.ny, 0);
        write_decay_row(decay_file, 0, 0.0, initial_sim, initial_theory);

        lbm_d2q9::write_scalar_csv(make_output_name(config.output_dir, "rho", 0), rho, config.nx, config.ny);
        lbm_d2q9::write_velocity_csv_pair(
            make_output_name(config.output_dir, "ux", 0),
            make_output_name(config.output_dir, "uy", 0),
            u,
            config.nx,
            config.ny);

        std::cout << "Step 0: a_sim = " << initial_sim << ", a_theory = " << initial_theory
                  << ", relative_error = "
                  << std::abs(initial_sim - initial_theory) / std::max(std::abs(initial_theory), 1e-14) << std::endl;

        for (int step = 1; step <= config.steps; ++step) {
            lbm_d2q9::streaming(f, f_next, config.nx, config.ny);
            lbm_d2q9::compute_density(f_next, rho, config.nx, config.ny);
            lbm_d2q9::compute_velocity(f_next, rho, u, config.nx, config.ny);
            lbm_d2q9::collide_bgk(f_next, rho, u, config.omega, config.nx, config.ny);
            Kokkos::fence();

            std::swap(f, f_next);

            const double a_sim = lbm_d2q9::measure_shear_wave_amplitude(u, config.nx, config.ny);
            const double a_theory =
                lbm_d2q9::analytical_shear_wave_amplitude(config.amplitude, nu_theory, config.ny, step);
            write_decay_row(decay_file, step, static_cast<double>(step), a_sim, a_theory);

            if (step % config.write_field_every == 0) {
                lbm_d2q9::write_scalar_csv(make_output_name(config.output_dir, "rho", step), rho, config.nx, config.ny);
                lbm_d2q9::write_velocity_csv_pair(
                    make_output_name(config.output_dir, "ux", step),
                    make_output_name(config.output_dir, "uy", step),
                    u,
                    config.nx,
                    config.ny);
            }

            if (step % config.write_field_every == 0) {
                const double relative_error = std::abs(a_sim - a_theory) / std::max(std::abs(a_theory), 1e-14);
                std::cout << "Step " << step << ": a_sim = " << a_sim << ", a_theory = " << a_theory
                          << ", relative_error = " << relative_error << std::endl;
            }
        }
    }
    Kokkos::finalize();
    return 0;
}
