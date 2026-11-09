cmake_minimum_required(VERSION 3.26)
project(pvalues_fourier C)

find_package(PkgConfig REQUIRED)
pkg_check_modules(SIMDE REQUIRED IMPORTED_TARGET simde)

set(GCC_COVERAGE_COMPILE_FLAGS "-Wall")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )
set(CMAKE_C_STANDARD 17)
SET(CMAKE_C_FLAGS_RELEASE "-Ofast -fno-finite-math-only ${CMAKE_C_FLAGS_RELEASE}")


add_executable(pvalues_fourier
        src/main.c
        src/multinomial_cl/kernel.cl
)
target_link_libraries(pvalues_fourier
        PkgConfig::SIMDE
        "-framework OpenCL"
        quadmath
        ${CMAKE_SOURCE_DIR}/ExactMultinom/exact_multinom.so
        ${CMAKE_SOURCE_DIR}/BagFFT/bagfft.so
)

set_target_properties(pvalues_fourier PROPERTIES
        BUILD_RPATH "${CMAKE_SOURCE_DIR}/ExactMultinom;${CMAKE_SOURCE_DIR}/BagFFT"
)

add_custom_command(TARGET pvalues_fourier POST_BUILD
        COMMAND install_name_tool -change ExactMultinom/exact_multinom.so @rpath/exact_multinom.so $<TARGET_FILE:pvalues_fourier>
        COMMAND install_name_tool -change BagFFT/bagfft.so @rpath/bagfft.so $<TARGET_FILE:pvalues_fourier>
        COMMENT "Rewriting bundled .so install names to @rpath"
)
