# https://mesonbuild.com/
project(
    'my-fantastic-simulation-code', # Project name
    ['c', 'cpp'], # Project type. We need a C++ compiler. The C compiler is required for MPI.
    version : '0.1'
)

cmake = import('cmake')
add_project_arguments('-std=c++17', language : ['cpp'])

# Optional dependency: Eigen3 linar algebra library
# Make sure to add 'eigen' to the dependencies in your subdirectories if you
# enable this.
# eigen = dependency('eigen3', version: '>=3.4.0')

# Optional dependency: Kokkos C++ performance portability library
# Make sure to add 'kokkoscore' to the dependencies in your subdirectories if you
# enable this.
kokkos = dependency('kokkoscore', required : false)

if not kokkos.found()
    kokkos_backend = get_option('kokkos_backend')
    kokkos_cuda_arch = get_option('kokkos_cuda_arch')

    kokkos_options = cmake.subproject_options()

    if kokkos_backend == 'cuda'
        kokkos_options.add_cmake_defines({
            'Kokkos_ENABLE_CUDA' : 'ON',
            'Kokkos_ENABLE_SERIAL' : 'ON',
            'Kokkos_ENABLE_OPENMP' : 'OFF',
            'Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE' : 'ON',
        })
        if kokkos_cuda_arch == 'AMPERE80'
            kokkos_options.add_cmake_defines({'Kokkos_ARCH_AMPERE80' : 'ON'})
        elif kokkos_cuda_arch == 'HOPPER90'
            kokkos_options.add_cmake_defines({'Kokkos_ARCH_HOPPER90' : 'ON'})
        endif
    else
        kokkos_options.add_cmake_defines({
            'Kokkos_ENABLE_CUDA' : 'OFF',
            'Kokkos_ENABLE_SERIAL' : 'ON',
            'Kokkos_ENABLE_OPENMP' : 'OFF',
        })
    endif

    kokkos_proj = cmake.subproject('kokkos', options : kokkos_options)
    kokkos = kokkos_proj.dependency('kokkoscore')
endif

# Detect MPI, but make it optional
mpi = dependency('mpi', language: 'cpp', required: false)

if mpi.found()
    add_project_arguments('-DWITH_MPI', language : ['c', 'cpp'])
    message('MPI found')
else
    message('MPI not found, disabling MPI support')
endif

subdir('src')
subdir('executables')
if get_option('enable_tests')
    subdir('tests')
endif
