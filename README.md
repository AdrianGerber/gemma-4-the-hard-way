# gemma-4-the-hard-way

## Objectives

Create a C program that can run inference using one of Google's new [Gemma 4](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/) model(s). Dependencies other than the standard library and operating system headers must be avoided. The primary objective is learning more about LLM architecture and what it takes to run such models.

### Rules / Concept / Specification:

- Create a command line application that can run LLM inference.
- Weights are parsed from a GGUF file.
- This project is about learning the details. AI coding assistants and agents may not be used to generate any part of the code except for unit tests.
- Performance, compatibility, portability and reusability are not priorities.
- No pre-existing code, except for the C standard library, operating system headers for mmap and a unit testing framework are used. I know that this means "re-inventing the wheel" and that's the whole point of this project.

## Usage

See VSCode configure and build tasks.

```sh
# Configure
gemma-4-the-hard-way$ cmake -Bbuild .
# Build
gemma-4-the-hard-way$ cmake --build build
# Test
gemma-4-the-hard-way$ ctest --test-dir build

# Download model
gemma-4-the-hard-way$ wget https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-e2b-it-Q8_0.gguf
```

## References

Any accessed external documentation, tutorials and reference material will be declared in this section.

- [Unity Test Framework](https://github.com/ThrowTheSwitch/Unity/blob/master). MIT License. Code from their `src` directory was copied directly to `tests/framework` in this repository.
- [GGUF Format Specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md). Enums were copied from the pseudocode.
- [Gemma 4 Model on Hugging Face](https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF)
- [Hugging Face LLM Course (Tokenizers)](https://huggingface.co/learn/llm-course/en/chapter2/4)  
- [SentencePiece Tokenizer Documentation](https://github.com/google/sentencepiece)
- [A Visual Guide to Gemma 4](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4)
- [Float16 De-Quantization from libcanard](https://github.com/OpenCyphal/libcanard/blob/636795f4bc395f56af8d2c61d3757b5e762bb9e5/canard.c#L811-L834)
