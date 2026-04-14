# Library with core logic
add_library(gemma_lib GGUF.c Tokenizer.c Model.c)
target_include_directories(gemma_lib PUBLIC .)
