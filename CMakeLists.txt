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
        -O2 -g
        -Wundef
        -Wstrict-prototypes
        -Wmissing-prototypes
        # -fanalyzer --> fanalyzer really doesn't seem to like returning pointers to malloc'd data in a struct by value. My code may well have other memory leaks, but I don't think fanalyzer is correct there.
        # -fsanitize=address --> I disabled this because of the performance impact, but at the moment it doesn't find any leaks or out-of-bounds accesses when enabled.
    )
    # add_link_options(
    #     -fsanitize=address
    # )
endif()

# Add source directory
add_subdirectory(src)

# Main executable
add_executable(gemma_4_the_hard_way src/main.c)
target_link_libraries(gemma_4_the_hard_way PRIVATE gemma_lib m)

