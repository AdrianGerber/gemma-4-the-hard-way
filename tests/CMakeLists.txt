# Unity framework library
add_library(unity_lib framework/unity.c)
target_include_directories(unity_lib PUBLIC framework)

# Unit test for file2
add_executable(test_file2 test_file2.c)
target_link_libraries(test_file2 PRIVATE gemma_lib unity_lib)

# Register the test with ctest
add_test(NAME test_file2 COMMAND test_file2)
