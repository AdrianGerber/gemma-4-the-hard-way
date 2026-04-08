# Unity framework library
add_library(unity_lib framework/unity.c)
target_include_directories(unity_lib PUBLIC framework)

# Unit test for GGUF methods
add_executable(test_gguf test_gguf.c)
target_link_libraries(test_gguf PRIVATE gemma_lib unity_lib)

# Register the test with ctest
add_test(NAME test_gguf COMMAND test_gguf)
