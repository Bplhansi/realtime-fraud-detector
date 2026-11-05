# Build instructions for the C++ scoring engine.
cmake_minimum_required(VERSION 3.20)
project(fraud_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # the Python module links the same library

# Single-config generators (Makefiles/Ninja on Linux) build unoptimised if no type is
# given; default to Release so latency numbers never come from a slow build by accident.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(MSVC)
    set(FD_OPT_FLAG "/O2")
    set(FD_WARNINGS /W4 /permissive-)
else()
    set(FD_OPT_FLAG "-O3")
    set(FD_WARNINGS -Wall -Wextra -Wpedantic)
endif()
add_compile_options($<$<CONFIG:Release>:${FD_OPT_FLAG}>)

# --- Engine library: the real code lives here -------------------------------
add_library(fd_core STATIC
    src/features.cpp
    src/latency.cpp
    src/model.cpp
    src/scorer.cpp
    src/thresholds.cpp
    src/transaction.cpp
)
target_include_directories(fd_core PUBLIC include)
target_compile_options(fd_core PRIVATE ${FD_WARNINGS})

# --- ONNX Runtime: runs the trained model inside the engine ------------------
# Downloads Microsoft's prebuilt release, pinned by version and SHA-256 so a changed file
# fails the build. Configure with -DFD_WITH_ONNX=OFF to build without model scoring.
include(FetchContent)
option(FD_WITH_ONNX "Score transactions with an ONNX model via ONNX Runtime" ON)
set(FD_ORT_DLLS "")
if(FD_WITH_ONNX)
    set(FD_ORT_VERSION 1.31.1)
    set(_ort_base "https://github.com/microsoft/onnxruntime/releases/download/v${FD_ORT_VERSION}")
    if(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(_ort_url "${_ort_base}/onnxruntime-win-x64-${FD_ORT_VERSION}.zip")
        set(_ort_sha 5869e26bb25314375f2d5ef1877e3ea06b1543d6c710e6e4a1ec2f7177ab4b2c)
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        set(_ort_url "${_ort_base}/onnxruntime-linux-x64-${FD_ORT_VERSION}.tgz")
        set(_ort_sha c849e36a1ef3e22d4ff217b9d1ce4a3cf5b9467b975610913ae959ad5b16bdc9)
    else()
        message(FATAL_ERROR "No prebuilt ONNX Runtime for this platform; use -DFD_WITH_ONNX=OFF")
    endif()
    FetchContent_Declare(onnxruntime URL "${_ort_url}" URL_HASH SHA256=${_ort_sha}
                         DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(onnxruntime)  # just unpacks: the release has no CMakeLists
    add_library(onnxruntime_prebuilt SHARED IMPORTED)
    set_target_properties(onnxruntime_prebuilt PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/include")
    if(WIN32)
        set_target_properties(onnxruntime_prebuilt PROPERTIES
            IMPORTED_IMPLIB "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.lib"
            IMPORTED_LOCATION "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.dll")
        # Windows has an older onnxruntime.dll in System32; ours must sit next to each program.
        set(FD_ORT_DLLS "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.dll"
                        "${onnxruntime_SOURCE_DIR}/lib/onnxruntime_providers_shared.dll")
    else()
        set_target_properties(onnxruntime_prebuilt PROPERTIES
            IMPORTED_LOCATION "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so")
    endif()
    target_link_libraries(fd_core PUBLIC onnxruntime_prebuilt)
    target_compile_definitions(fd_core PUBLIC FD_WITH_ONNX=1)
endif()

# Copies the ONNX Runtime DLLs next to a program on Windows (does nothing elsewhere).
function(fd_copy_runtime_dlls target)
    if(FD_ORT_DLLS)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FD_ORT_DLLS} $<TARGET_FILE_DIR:${target}>)
    endif()
endfunction()

# --- Replay program -----------------------------------------------------------
add_executable(fd_replay src/replay_main.cpp)
target_link_libraries(fd_replay PRIVATE fd_core)
target_compile_options(fd_replay PRIVATE ${FD_WARNINGS})
target_compile_definitions(fd_replay PRIVATE
    FD_BUILD_CONFIG="$<CONFIG>"
    FD_BUILD_OPT="$<$<CONFIG:Release>:${FD_OPT_FLAG}>")
fd_copy_runtime_dlls(fd_replay)

# --- Python module (pybind11): the API calls the engine through it ----------------
# Python is also needed for the parity tests. pybind11 comes from pip (requirements.txt),
# so no extra download is needed.
option(FD_WITH_PYTHON "Build the fraud_engine Python module" ON)
if(FD_WITH_PYTHON)
    find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
else()
    find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter)
endif()
if(FD_WITH_PYTHON)
    execute_process(COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
                    OUTPUT_VARIABLE _pybind11_dir OUTPUT_STRIP_TRAILING_WHITESPACE
                    RESULT_VARIABLE _pybind11_found)
    if(_pybind11_found EQUAL 0)
        find_package(pybind11 CONFIG REQUIRED PATHS "${_pybind11_dir}" NO_DEFAULT_PATH)
        pybind11_add_module(fraud_engine src/python_module.cpp)
        target_link_libraries(fraud_engine PRIVATE fd_core)
        # Put the module (and the ONNX Runtime DLLs on Windows) in engine/build/python,
        # whatever the build type, so the API knows where to look.
        set_target_properties(fraud_engine PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
            LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/python"
            LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/python")
        fd_copy_runtime_dlls(fraud_engine)
    else()
        message(WARNING "pybind11 not found for ${Python3_EXECUTABLE} (pip install pybind11); "
                        "skipping the Python module")
    endif()
endif()

# --- Tests ------------------------------------------------------------------
enable_testing()

# GoogleTest, downloaded at configure time (pinned release, checked by SHA-256).
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
    URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)  # same C runtime as our code on Windows
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(fd_tests
    tests/test_features.cpp
    tests/test_latency.cpp
    tests/test_model.cpp
    tests/test_parity.cpp
    tests/test_ring_buffer.cpp
    tests/test_scorer.cpp
    tests/test_thresholds.cpp
    tests/test_transaction.cpp
)
target_link_libraries(fd_tests PRIVATE fd_core GTest::gtest_main)
target_compile_options(fd_tests PRIVATE ${FD_WARNINGS})
fd_copy_runtime_dlls(fd_tests)

# Parity check: before the C++ tests run, Python simulates a few thousand transactions and
# writes them with Python's features, a small ONNX model and Python's scores. The C++ tests
# must reproduce them. Needs requirements.txt installed for this Python
# (locally: configure with -DPython3_EXECUTABLE=.venv/...).
set(FD_PARITY_DIR "${CMAKE_BINARY_DIR}/parity")
target_compile_definitions(fd_tests PRIVATE
    FD_PARITY_DIR="${FD_PARITY_DIR}"
    FD_FEATURES_TXT="${PROJECT_SOURCE_DIR}/../config/features.txt"
    FD_TEST_TMP="${CMAKE_BINARY_DIR}/test-tmp")
add_test(
    NAME python_writes_parity_files
    COMMAND Python3::Interpreter -m ml.export_onnx --parity-dir "${FD_PARITY_DIR}"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/.."
)
set_tests_properties(python_writes_parity_files PROPERTIES FIXTURES_SETUP parity)

include(GoogleTest)
# PRE_TEST: list the tests when CTest runs, after the ONNX Runtime DLLs have been copied.
gtest_discover_tests(fd_tests DISCOVERY_MODE PRE_TEST PROPERTIES FIXTURES_REQUIRED parity)

# Smoke tests for the replay program itself.
add_test(NAME fd_replay_prints_latency
         COMMAND fd_replay "${FD_PARITY_DIR}/parity.tx.bin" --repeat 2)
set_tests_properties(fd_replay_prints_latency PROPERTIES
    FIXTURES_REQUIRED parity
    PASS_REGULAR_EXPRESSION "throughput: [0-9]+ tx/s.*latency p50: [0-9]+ ns.*latency p99: [0-9]+ ns")
add_test(NAME fd_replay_rejects_missing_file COMMAND fd_replay "does-not-exist.tx.bin")
set_tests_properties(fd_replay_rejects_missing_file PROPERTIES
    PASS_REGULAR_EXPRESSION "cannot read transaction file")
add_test(NAME fd_replay_rejects_bad_repeat COMMAND fd_replay "x.tx.bin" --repeat 2.5)
set_tests_properties(fd_replay_rejects_bad_repeat PROPERTIES
    PASS_REGULAR_EXPRESSION "--repeat needs a whole number.*usage: fd_replay")
add_test(NAME fd_replay_needs_thresholds_with_model COMMAND fd_replay "x.tx.bin" --model m.onnx)
set_tests_properties(fd_replay_needs_thresholds_with_model PROPERTIES
    PASS_REGULAR_EXPRESSION "--model and --thresholds go together")
add_test(NAME fd_replay_reports_build_type
         COMMAND fd_replay "${FD_PARITY_DIR}/parity.tx.bin")
set_tests_properties(fd_replay_reports_build_type PROPERTIES
    FIXTURES_REQUIRED parity
    PASS_REGULAR_EXPRESSION "build: (Release|Debug|RelWithDebInfo|MinSizeRel)")
