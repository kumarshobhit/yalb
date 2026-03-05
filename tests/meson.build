google_test = dependency('gtest')

test_sources = [
    'test_hello_world.cpp',
    'test_lbm_d2q9.cpp',
    'test_lbm_collision.cpp',
    'test_lbm_shear_wave.cpp'
]

if mpi.found()
    test_sources += [
        'gtest_mpi_main.cpp'
    ]
else
    test_sources += [
        'gtest_main.cpp'
    ]
endif

tests = executable(
    'tests',
    test_sources,
    include_directories : [lib_incdirs],
    link_with : [lib],
    dependencies : [google_test, kokkos, mpi]
)

test('Tests', tests)
