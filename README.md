# Meson skeleton code

This repository contains a [Meson](https://mesonbuild.com/) skeleton for C++ projects. It has provision
for dependencies on the numerical libraries

* [Eigen3](https://eigen.tuxfamily.org/)
* [Kokkos](https://kokkos.org/)
* [MPI](https://www.mpi-forum.org/)

## Getting started

Click the `Use this template` button above, then clone the newly created
repository.

### Compiling using CLion

> Note: for Windows users, please follow [these
> instructions](https://www.jetbrains.com/help/clion/how-to-use-wsl-development-environment-in-product.html)
> in addition to the text below.

If you are using CLion, you can open your freshly cloned project by clicking on
the "Open" button in the CLion welcome window. If prompted, trust the project.

You can change Meson build option in "**File > Settings > Build, Execution, Deployment > Meson**".
This windows allows to set the `buildtype` option, which controls the level of optimization applied to
the code. `debug` disables optimizations and turns on useful debugging features.
This mode should be used when developing and testing code.
`release` turns on aggressive optimization. This mode should be used when
running production simulations. Add `--buildtype=release` or `--buildtype=debug` to "Setup options"
to switch between the two.

To run the executable, click on the dialog directly right of the
green hammer in the upper right toolbar, select "main", and click
the green arrow right of that dialog. You should see the output in the "Run"
tab, in the lower main window.

To run the tests, select "tests" in the same dialog, then run. In the lower
window, on the right, appears a panel that enumerates all the tests that were
run and their results.

Try compiling and running for both `debug` and `release` configurations. Don't
forget to switch between them when testing code or running production simulations.

### Compiling from the command line

The command line (terminal) may look daunting at first, but it has the advantage
of being the same across all UNIX platforms, and does not depend on a specific
IDE. The standard Meson workflow is to create a `builddir/` directory which will
contain all the build files. To do that, and compile your project, run:

```bash
cd <your repository>

# Configure and create build directory
meson setup builddir

# Compile
cd builddir
meson compile

# Run executable and tests
./main
meson test
```

Note that CLion is by default configured to create a `buildDir/` directory
(with a capital `D`).

If there are no errors then you are all set! Note that the flag
`--buildtype=debug` should be changed to
`--buildtype=release` when you run a production simulation, i.e. a
simulation with more than a few hundred atoms. This turns on aggressive compiler
optimizations, which results in speedup. However, when writing the code and
looking for bugs, `debug` should be used instead.

Try compiling and running tests with both compilation configurations.

### Running milestone02

The repository now includes a `milestone02` executable that implements the
streaming-only D2Q9 lattice Boltzmann milestone with Kokkos.

Build it from the command line with:

```bash
meson compile -C builddir milestone02
```

Run it with the default setup:

```bash
cd builddir
./executables/milestone02/milestone02
```

This writes density and velocity snapshots to `builddir/outputs/`:

- `rho_step_000.csv`
- `ux_step_000.csv`
- `uy_step_000.csv`

The executable accepts a few parameters so you can experiment without editing
the source:

```bash
./executables/milestone02/milestone02 \
  --nx 20 \
  --ny 12 \
  --steps 15 \
  --x 3 \
  --y 4 \
  --direction 1 \
  --value 1.0 \
  --output-dir outputs/custom_run
```

Options:

- `--nx`, `--ny`: grid width and height
- `--steps`: number of streaming steps
- `--x`, `--y`: initial packet position
- `--direction`: initial D2Q9 direction index from `0` to `8`
- `--value`: initial packet magnitude
- `--output-dir`: output folder for CSV files

To generate plots from the output files, run:

```bash
cd builddir
python3 ../executables/milestone02/plot.py
```

The plotting script reads `outputs/rho_step_*.csv` and, if present,
`outputs/ux_step_*.csv` plus `outputs/uy_step_*.csv`, then writes PNGs into
`outputs/plots/`.

### Running milestone03

The repository also includes a `milestone03` executable that adds the BGK
collision operator on top of the D2Q9 streaming step.

Build it from the command line with:

```bash
meson compile -C builddir executables/milestone03/milestone03
```

Run the default "bump" scenario:

```bash
cd builddir
./executables/milestone03/milestone03 --steps 50 --output-dir outputs_m3_bump
```

This initializes a uniform density field with a slightly higher density at the
 center of the domain and zero initial velocity everywhere.

Run the "drift" scenario:

```bash
cd builddir
./executables/milestone03/milestone03 \
  --steps 50 \
  --scenario drift \
  --output-dir outputs_m3_drift
```

This initializes a localized higher-density region with a small positive
 x-velocity so you can observe transport and relaxation together.

Useful options:

- `--nx`, `--ny`: grid width and height
- `--steps`: number of time steps
- `--omega`: BGK relaxation parameter, must satisfy `0 < omega < 2`
- `--scenario`: either `bump` or `drift`
- `--base-rho`: background density
- `--bump-rho`: higher density used for the center bump or drifting blob
- `--drift-ux`: x-velocity used in the `drift` scenario
- `--output-dir`: output folder for CSV files

To generate plots from the Milestone 03 output files, run:

```bash
cd builddir
python3 ../executables/milestone03/plot.py --input-dir outputs_m3_bump
python3 ../executables/milestone03/plot.py --input-dir outputs_m3_drift
```

The plotting script reads `rho_step_*.csv`, `ux_step_*.csv`, and
`uy_step_*.csv`, then writes PNGs into `<output-dir>/plots/`.

### Running milestone04

Milestone 4 validates the BGK D2Q9 solver with a 2D shear-wave decay benchmark.

Build it from the command line with:

```bash
meson compile -C builddir executables/milestone04/milestone04
```

Run the benchmark:

```bash
cd builddir
./executables/milestone04/milestone04 \
  --nx 64 \
  --ny 64 \
  --steps 200 \
  --omega 1.0 \
  --rho0 0.2 \
  --amplitude 0.05 \
  --output-dir outputs_m4
```

This writes:

- `shear_decay.csv` with columns
  - `step,time,a_sim,a_theory,relative_error`
- field snapshots (`rho_step_###.csv`, `ux_step_###.csv`, `uy_step_###.csv`)
  every `--write-field-every` steps

Useful options:

- `--omega`: BGK relaxation parameter (`0 < omega < 2`)
- `--amplitude`: initial shear-wave amplitude (`0 <= amplitude < 0.1`)
- `--write-field-every`: output interval for field snapshots
- `--output-dir`: destination folder

Generate validation plots:

```bash
cd builddir
python3 ../executables/milestone04/plot.py --input-dir outputs_m4
```

Generated files include:

- `plots/shear_decay.png` (simulated vs analytical amplitude decay)
- `plots/relative_error.png`
- `plots/velocity_profile_step_###.png` (x-averaged velocity profiles with fixed y-scale)
- `plots/velocity_profile_overlay.png` (velocity profiles over multiple steps)
- `plots/density_profile_step_###.png` (x-averaged density profiles with fixed y-scale)
- `plots/density_profile_overlay.png` (density profiles over multiple steps)

The script also writes `plots/profile_overlay.png` for backward compatibility.

Measure viscosity as a function of `omega` and compare to analytical theory:

```bash
cd <repo>
python3 executables/milestone04/measure_viscosity_vs_omega.py \
  --build-dir build_milestone \
  --nx 64 \
  --ny 64 \
  --steps 200 \
  --rho0 0.2 \
  --amplitude 0.05 \
  --omegas 0.6,0.8,1.0,1.2,1.4,1.6 \
  --output-dir build_milestone/outputs_m4_sweep
```

This produces:

- `build_milestone/outputs_m4_sweep/viscosity_vs_omega.csv`
- `build_milestone/outputs_m4_sweep/viscosity_vs_omega.png`

The CSV columns are:

- `omega`
- `nu_measured`
- `nu_theory`
- `relative_error`
- `fit_r2`

### Running milestone05

Milestone 5 simulates the 2D D2Q9 lid-driven cavity with no-slip walls on
left/right/bottom and a moving top lid.

Build it from the command line with:

```bash
meson compile -C builddir executables/milestone05/milestone05
```

Run with defaults:

```bash
cd builddir
./executables/milestone05/milestone05
```

Run with explicit settings (report-ready setup):

```bash
cd builddir
./executables/milestone05/milestone05 \
  --nx 128 \
  --ny 128 \
  --steps 20000 \
  --omega 1.0 \
  --rho0 1.0 \
  --lid-ux 0.05 \
  --lid-uy 0.0 \
  --write-field-every 1000 \
  --residual-every 1000 \
  --output-dir outputs_m5
```

Useful options:

- `--nx`, `--ny`: grid dimensions (`> 2`)
- `--steps`: number of time steps (`>= 0`)
- `--omega`: BGK relaxation parameter (`0 < omega < 2`)
- `--rho0`: initial density and wall density used in moving-wall correction (`> 0`)
- `--lid-ux`, `--lid-uy`: top-wall velocity components (`|u| < 0.1`)
- `--write-field-every`: output interval for `rho/ux/uy` snapshots
- `--residual-every`: output interval for residual report (`max_delta_u`)
- `--output-dir`: output folder

Generated simulation artifacts include:

- `residual_history.csv` (`step,max_delta_u`)
- `rho_step_#####.csv`, `ux_step_#####.csv`, `uy_step_#####.csv`

Generate Milestone 05 plots:

```bash
cd builddir
python3 ../executables/milestone05/plot.py --input-dir outputs_m5
```

This writes into `outputs_m5/plots/`:

- `velocity_quiver_last.png`
- `streamlines_last.png`
- `speed_contour_last.png`
- `u_centerline_vertical_last.png`
- `v_centerline_horizontal_last.png`
- `velocity_profile_overlay.png`
- `density_profile_overlay.png`
- `residual_history.png`

### Running milestone06

Milestone 6 parallelizes the lid-driven cavity solver across MPI ranks with
1D x-direction domain decomposition and one-column halo exchange.

Communication pattern and halo width (implemented):

- Decomposition: 1D slabs in x, each rank owns a contiguous x-range.
- Halo width: 1 column on each side of the local slab.
- Exchanged data: full D2Q9 distribution `f(x, y, direction)` for all
  `direction = 0..8` and all local `y`.
- Neighbor pattern: left/right only (no y-neighbor exchange in this 1D split).
- MPI primitive: `MPI_Sendrecv` in two pairs to avoid deadlock.

Conceptual layout for rank `r`:

```text
left ghost | owned columns              | right ghost
  x=0      | x=1 ... x=local_nx         | x=local_nx+1
           ^ send to left from x=1
           ^ recv from left into x=0
                                   ^ send to right from x=local_nx
                                   ^ recv from right into x=local_nx+1
```

Build it (only available when MPI is found during Meson configure):

```bash
meson compile -C builddir executables/milestone06/milestone06
```

Run on 4 ranks:

```bash
cd builddir
mpirun -np 4 ./executables/milestone06/milestone06 \
  --nx 128 \
  --ny 128 \
  --steps 20000 \
  --omega 1.0 \
  --rho0 1.0 \
  --lid-ux 0.05 \
  --lid-uy 0.0 \
  --write-field-every 1000 \
  --residual-every 1000 \
  --output-dir outputs_m6
```

Residual diagnostics are written to `residual_history.csv` with columns:

- `step`
- `max_delta_u`
- `global_mass`
- `global_ke`

Generate Milestone 06 plots from gathered global fields:

```bash
cd builddir
python3 ../executables/milestone06/plot.py --input-dir outputs_m6
```

Generated plot files include:

- `velocity_quiver_last.png`
- `streamlines_last.png`
- `speed_contour_last.png`
- `u_centerline_vertical_last.png`
- `v_centerline_horizontal_last.png`
- `velocity_profile_overlay.png`
- `density_profile_overlay.png`
- `residual_history.png`
- `global_mass_history.png`
- `global_ke_history.png`

Measure strong scaling:

```bash
cd <repo>
python3 executables/milestone06/measure_scaling.py \
  --build-dir builddir \
  --output-dir builddir/outputs_m6_scaling \
  --nprocs 1,2,4 \
  --nx 128 \
  --ny 128 \
  --steps 2000
```

This writes:

- `builddir/outputs_m6_scaling/scaling.csv`
- `builddir/outputs_m6_scaling/speedup_vs_np.png`

GPU strong scaling via CMake (Kokkos CUDA on A100):

```bash
sbatch executables/milestone06/slurm/gpu_strong_scaling_cmake_a100.slurm
```

This script builds Kokkos + the Milestone 06 CMake target and writes logs/plots to `outputs_gpu`.

### Compiling on bwUniCluster, with MPI

The above steps should be done *after* loading the appropriate packages:

```bash
module load compiler/gnu mpi/openmpi

# then in build/
meson setup builddir --buildtype=release
cd builddir
meson compile
```

## How to add code to the repository

There are three places where you are asked to add code:

- `src/` is the core of the code. Code common to all executables and tests
  you will run should be added here. The `meson.build` file in `src/` creates
  [static library](https://en.wikipedia.org/wiki/Static_library) which is linked
  to all the other targets in the repository, and which propagates its dependency,
  so that there is no need to explicitly link against Eigen or MPI.
- `tests/` contains tests for the library code code. It uses
  [GoogleTest](https://google.github.io/googletest/) to define short, simple
  test cases.
- `executables/` contains the final executable codes, i.e. it needs a `main()`
  function.

### Adding to `src/`

Adding files to `src/` is straightforward: create your files, e.g. `lj.h` and
`lj.cpp`, then update the `lib_sources` variable in
`meson.build`:

```meson
lib_sources = [  # All source files (excluding headers)
    'hello.cpp',
    'lj.cpp'
]
```

### Adding to `tests/`

Create your test file, e.g. `test_verlet.cpp` in `tests/`, then modify the
`test_sources` variable in `tests/meson.build`. Test that your test was
correctly added by running `meson compile` in the build directory: your test should
show up in the output.

### Adding to `executables/`

Create a new directory, e.g. with `mkdir executables/04`, then add `subdir('04')`
to `executables/meson.build`, then create & edit
`executables/04/meson.build`:

```meson
executable(
    'milestone04',
    'main.cpp',
    include_directories : [lib_incdirs],
    link_with : [lib],
    dependencies : [eigen, mpi]
)
```

You can now create & edit `executables/04/main.cpp`, which should include a
`main()` function as follows:

```c++
int main(int argc, char* argv[]) {
    return 0;
}
```

The code of your simulation goes into the `main()` function.

#### Input files

We often provide input files (`.xyz` files) for your simulations, for example in
milestone 4. You should place these in e.g. `executables/04/`, and add the
following to `executables/04/meson.build`:

```meson
fs = import('fs')
fs.copyfile('lj54.xyz')
```

This will copy the file `executables/04/lj54.xyz` to
`<build>/executables/04/lj54.xyz`, but **only** when the executable for the milestone
is rebuilt. To trigger a rebuild you can erase the `<build>/executables/04`
directory and `meson compile` again.

*Note:* `.xyz` files are ignored by Git. That's on purpose to avoid you staging
very large files in the git tree.

## Pushing code to GitHub

If you have added files to your local repositories, you should commit and push them to
GitHub. To create a new commit (i.e. put your files in the repository's
history), simply run:

```bash
git status
# Look at the files that need to be added
git add <files> ...
git commit -m '<a meaningful commit message!>'
git push
```

This repository is setup with continuous integration (CI), so all your tests
will run automatically when you push. This is very handy to test if you have
breaking changes.

### Git in CLion

If you are using CLion, you can use Git directly from its interface. To add
files, right click the file you wish to add, then "Git > Add". Once you are
ready to commit, "Git > Commit" from the main menu bar. Add a message in the
lower left window where it reads "Commit message", then click "Commit" or
"Commit and Push...".
