cmake_minimum_required(VERSION 3.12)
project(CRYPTOFL LANGUAGES CXX)

if (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0)
    message(FATAL_ERROR "ABY requires at least g++-8")
endif ()


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/ABY/cmake")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

# Set build type to `Release` if non was specified:
# (cf. https://gitlab.kitware.com/cmake/community/wikis/FAQ#how-can-i-change-the-default-build-mode-and-see-it-reflected-in-the-gui)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
            "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
            FORCE)
endif (NOT CMAKE_BUILD_TYPE)

# Write built executables and libraries to bin/ and lib/, respectively.
if (NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
endif ()
if (NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
endif ()
if (NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
endif ()

find_package(ENCRYPTO_utils QUIET)
if (ENCRYPTO_utils_FOUND)
    message(STATUS "Found ENCRYPTO_utils")
elseif (NOT ENCRYPTO_utils_FOUND AND NOT TARGET ENCRYPTO_utils::encrypto_utils)
    message("ENCRYPTO_utils was not found: add ENCRYPTO_utils subdirectory")
    if (NOT EXISTS "${PROJECT_SOURCE_DIR}/ABY/extern/ENCRYPTO_utils/CMakeLists.txt")
        find_package(Git REQUIRED)
        message("initialize Git submodule: extern/ENCRYPTO_utils")
        execute_process(COMMAND git submodule update --init extern/ENCRYPTO_utils
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
    endif ()
    add_subdirectory(ABY/extern/ENCRYPTO_utils)
endif ()

find_package(OTExtension QUIET)
if (OTExtension_FOUND)
    message(STATUS "Found OTExtension")
elseif (NOT OTExtension_FOUND AND NOT TARGET OTExtension::otextension)
    message("OTExtension was not found: add OTExtension subdirectory")
    if (NOT EXISTS "${PROJECT_SOURCE_DIR}/ABY/extern/OTExtension/CMakeLists.txt")
        find_package(Git REQUIRED)
        message("initialize Git submodule: extern/OTExtension")
        execute_process(COMMAND git submodule update --init extern/OTExtension
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
    endif ()
    add_subdirectory(ABY/extern/OTExtension)
endif ()

find_package(GMP REQUIRED)
find_package(Threads REQUIRED)
find_package(Boost 1.66.0 REQUIRED COMPONENTS thread system)

add_subdirectory(ABY/src/abycore)
add_subdirectory(pybind11)

pybind11_add_module(CryptoFL CryptoFL_lib/cryptofl.cpp CryptoFL_lib/cryptofl_aby.cpp)
target_link_libraries(CryptoFL PRIVATE ABY::aby)
