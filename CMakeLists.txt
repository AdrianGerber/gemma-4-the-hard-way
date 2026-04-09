cmake_minimum_required(VERSION 3.10)
project(Gemma4TheHardWay C)

set(CMAKE_C_STANDARD 11)

# Strict GCC compile flags
if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    add_compile_options(
        -Wall
        -Wextra
        -Wshadow
        -Wwrite-strings
        -Wsign-conversion
        -Werror
        -Wformat=2
        -Wundef
        -Wstrict-prototypes
        -Wmissing-prototypes
#        -fanalyzer --> fanalyzer really doesn't seem to like returning pointers to malloc'd data in a struct by value. My code may well have other memory leaks, but I don't think fanalyzer is correct here.    
    )
endif()

# Add source directory
add_subdirectory(src)

# Main executable
add_executable(hello_world src/main.c)
target_link_libraries(hello_world PRIVATE gemma_lib)

# Testing configuration
enable_testing()
add_subdirectory(tests)
