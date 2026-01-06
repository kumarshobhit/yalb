#include <Kokkos_Core.hpp>
#include <iostream>
#include <fstream>
#include <vector>

// --- D2Q9 Constants ---
// 9 Directions: Center (0), E, N, W, S, NE, NW, SW, SE
const int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
const int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};

// Grid Dimensions (Small for testing)
const int Nx = 15;
const int Ny = 10;
const int Q = 9; // 9 velocities

// Function to compute Density and Velocity at every cell
// Stores result in 'density' and 'velocity' views
void compute_moments(Kokkos::View<double***> f, 
                     Kokkos::View<double**> rho, 
                     Kokkos::View<double***> u, 
                     int Nx, int Ny) {
    
    Kokkos::parallel_for("ComputeMoments", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
    KOKKOS_LAMBDA(const int x, const int y) {
        double density = 0.0;
        double u_x = 0.0;
        double u_y = 0.0;

        // Sum up all 9 directions
        for (int i = 0; i < 9; ++i) {
            double val = f(x, y, i);
            density += val;             // Density = Sum of f
            u_x += val * cx[i];         // Velocity X = Sum of (f * direction_x)
            u_y += val * cy[i];         // Velocity Y = Sum of (f * direction_y)
        }

        // Store Density
        rho(x, y) = density;

        // Normalize Velocity (u = momentum / density)
        // Avoid division by zero!
        if (density > 1e-9) {
            u(x, y, 0) = u_x / density;
            u(x, y, 1) = u_y / density;
        } else {
            u(x, y, 0) = 0.0;
            u(x, y, 1) = 0.0;
        }
    });
}

void write_output(std::string filename, Kokkos::View<double**> rho, int Nx, int Ny) {
    auto rho_host = Kokkos::create_mirror_view(rho);
    Kokkos::deep_copy(rho_host, rho); // Copy from GPU/Device to CPU

    std::ofstream file(filename);
    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            file << rho_host(x, y);
            if (x < Nx - 1) file << ","; // CSV format
        }
        file << "\n";
    }
    file.close();
}

int main(int argc, char* argv[]) {
    // Initialize Kokkos (and MPI implicitly if enabled)
    Kokkos::initialize(argc, argv);
    {
        std::cout << "Initializing Milestone 2 Simulation..." << std::endl;

        // 1. Create Data Structures (Views)
        Kokkos::View<double***> f("f", Nx, Ny, Q);
        Kokkos::View<double***> f_next("f_next", Nx, Ny, Q);
        
        // NEW: Views for Density (rho) and Velocity (u)
        Kokkos::View<double**> rho("rho", Nx, Ny);
        Kokkos::View<double***> u("u", Nx, Ny, 2); // 2 components: x, y

        // 2. Initialize with some data
        auto f_host = Kokkos::create_mirror_view(f);
        
        // Set a dot of density 1.0 moving East (direction 1) at (5,5)
        f_host(5, 5, 1) = 1.0; 
        
        Kokkos::deep_copy(f, f_host);

        // 3. Time Stepping Loop
        int num_steps = 10;
        for (int step = 0; step < num_steps; ++step) {
            
            // --- STREAMING STEP ---
            Kokkos::parallel_for("Streaming", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
            KOKKOS_LAMBDA(const int x, const int y) {
                for (int i = 0; i < Q; ++i) {
                    // Reverse direction to pull from neighbor
                    int neighbor_x = x - cx[i];
                    int neighbor_y = y - cy[i];

                    // Periodic Boundary Conditions
                    if (neighbor_x < 0) neighbor_x += Nx;
                    if (neighbor_x >= Nx) neighbor_x -= Nx;
                    
                    if (neighbor_y < 0) neighbor_y += Ny;
                    if (neighbor_y >= Ny) neighbor_y -= Ny;

                    // Pull the value
                    f_next(x, y, i) = f(neighbor_x, neighbor_y, i);
                }
            });
            Kokkos::fence();

            // Update f for the next step
            Kokkos::deep_copy(f, f_next);

            // --- COMPUTE MOMENTS (Density & Velocity) ---
            compute_moments(f, rho, u, Nx, Ny);

            std::string filename = "output_" + std::to_string(step) + ".csv";
            write_output(filename, rho, Nx, Ny);
            std::cout << "Step " << step << ": Written to " << filename << std::endl;

            // --- PRINT POSITION ---
            // We need to copy 'rho' back to the CPU (Host) to read it with if-statements
            auto rho_host = Kokkos::create_mirror_view(rho);
            Kokkos::deep_copy(rho_host, rho);

            // Scan the grid to find where our particle is
            bool found = false;
            for(int x = 0; x < Nx; ++x) {
                for(int y = 0; y < Ny; ++y) {
                    if(rho_host(x, y) > 0.5) { // If density is high
                        std::cout << "Step " << step << ": Particle found at (" << x << ", " << y << ")" << std::endl;
                        found = true;
                    }
                }
            }
            if(!found) std::cout << "Step " << step << ": Particle lost (Mass Error!)" << std::endl;
        }
    }
    Kokkos::finalize();
    return 0;
}