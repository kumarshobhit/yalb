#ifndef LBM_D2Q9_MPI_H
#define LBM_D2Q9_MPI_H

#include "lbm_d2q9.h"

#ifdef WITH_MPI
#include <mpi.h>
#endif

#include <vector>

namespace lbm_d2q9 {

struct DomainDecomposition1D {
    int rank = 0;
    int size = 1;
    int global_nx = 0;
    int global_ny = 0;
    int local_nx = 0;
    int x_offset = 0;
    int left_rank = -1;
    int right_rank = -1;
};

#ifdef WITH_MPI
DomainDecomposition1D decompose_domain_1d_x(int nx, int ny, MPI_Comm comm);

void initialize_from_macroscopic_fields_local(
    const DensityView& rho,
    const VelocityView& u,
    DistributionView& f_with_ghost,
    const DomainDecomposition1D& decomp);

void exchange_halo_columns(DistributionView& f_owned_with_ghost, const DomainDecomposition1D& decomp, MPI_Comm comm);

void stream_with_cavity_boundaries_local(
    const DistributionView& f_in,
    DistributionView& f_out,
    const DomainDecomposition1D& decomp,
    double lid_ux,
    double lid_uy,
    double rho_wall);

void compute_density_local(
    const DistributionView& f_with_ghost,
    DensityView& rho,
    const DomainDecomposition1D& decomp);

void compute_velocity_local(
    const DistributionView& f_with_ghost,
    const DensityView& rho,
    VelocityView& u,
    const DomainDecomposition1D& decomp);

void collide_bgk_local(
    DistributionView& f_with_ghost,
    const DensityView& rho,
    const VelocityView& u,
    double omega,
    const DomainDecomposition1D& decomp);

void compute_global_mass_kinetic_energy(
    const DensityView& rho,
    const VelocityView& u,
    const DomainDecomposition1D& decomp,
    double& global_mass,
    double& global_ke,
    MPI_Comm comm);

double compute_global_max_delta_u(
    const VelocityView& previous,
    const VelocityView& current,
    const DomainDecomposition1D& decomp,
    MPI_Comm comm);

void gather_fields_to_root(
    const DensityView& rho_local,
    const VelocityView& u_local,
    DensityView& rho_global_root,
    VelocityView& u_global_root,
    const DomainDecomposition1D& decomp,
    int root,
    MPI_Comm comm);
#endif

}  // namespace lbm_d2q9

#endif  // LBM_D2Q9_MPI_H
