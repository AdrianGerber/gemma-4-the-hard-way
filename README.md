# gemma-4-the-hard-way

A C program that can run inference using Google's [gemma-4-e2b-it-Q8_0](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/) model. Dependencies other than the standard library and operating system headers must be avoided. The primary objective is learning more about LLM architecture and what it takes to run such models.

## Usage

See VSCode configure and build tasks.

### Configure CMake

```sh
gemma-4-the-hard-way$ cmake -Bbuild .
```

### Build

```sh
gemma-4-the-hard-way$ cmake --build build
```

### Download the Model

```sh
# This model was used for development but is no longer available for some reason.
gemma-4-the-hard-way$ wget https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-e2b-it-Q8_0.gguf

# This one works as well after adding support for bf16 quantized matrices. 
gemma-4-the-hard-way$ wget https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q8_0.gguf
```

### Run inference

```
gemma-4-the-hard-way$ ./build/gemma_4_the_hard_way "Please tell a joke about a large language model."
...
Generating Predictions:

Here are a few jokes about large language models, depending on the style you prefer:

**Option 1: Self-Aware/A Bit Meta**

> Why did the large language model break up with the search engine?
>
> Because it felt like their relationship was too reliant on repetitive queries and lacked any real *context*!

**Option 2: Poking Fun at the Output**

> A user asked an LLM to write a poem about a cat.
>
> The model responded with a 10,000-word epic detailing the cat's existential dread, its relationship with the sunbeam, and its philosophical observations on tuna. The user just wanted a haiku.

**Option 3: The Technical Joke**

> What do you call an LLM that can't stop generating tangents?
>
> A recursive loop with a strong sense of *flow*!

**Which one do you like best?** 😊<turn|>
```

## Project Details

### Idea / Specification:

- Create a command line application that can run LLM inference.
- Performance, compatibility, portability and reusability are not priorities.
- Weights are parsed from a GGUF file. The model file can be assumed to be valid. Hardening against malformed or malicious files is not required.
- This project is about learning the details. AI may be used for research, learning, review and debugging purposes, but not for code generation or completion.
- No pre-existing code, except for the C standard library and basic operating system headers (e.g. mmap) must be included. I know that this means "re-inventing the wheel" and that's the whole point of this project.
- No advanced error handling is needed. The program can use simple asserts to terminate on issues (e.g. dimension mismatch or failed malloc). This means that it is not safe to run the code with asserts disabled.

### Lessons Learned

- I found the attention layer (including the KV cache) challenging to understand and implement. At least now I understand why context size is so computationally expensive :).
- C is not the most comfortable language for tasks where everything is as painfully dynamic as in machine learning. On the other hand, it would allow you to really optimize the code and buffers if you wanted to.  
- It would've been easier to start with a simpler language model. While gemma-4-e2b-it-Q8_0 is suprisingly powerful, it has some architectural specialties that I had to find out about the hard way :). I initially got very confused by the cache reuse and the alternating global / SWA layers.
- llama.cpp has great debugging options. It became my main way of identifying mistakes (e.g. misunderstood model structure / data flow, math errors, ...). Comparing against `llama-debug -m gemma-4-e2b-it-Q8_0.gguf -p "<bos>\n"` was very valuable.
- I initially set this project up with unit tests, but it turned out that these didn't help that much and so I deleted them. In my opinion, unit tests are essential when building and maintaining projects. However, using them in a way that makes sense here is non-trivial. Of course I could test my math primitives in different cases and this would likely also have helped find a few bugs, but the project's main challenge was figuring out the overall data flow and model structure - and it's very difficult to come up with test cases if you are dealing with huge matrices and don't even know the expected result. I guess the llama-debug traces ended up as a sort of "manual" unit tests. Maybe I could've automated this in some way.
- Having to parse the GGUF file and build the tokenizer added maybe 30% of complexity to the project. However, it was very useful in understanding the different data types and getting to know the basic model structure.
- Not focusing on performance was a good idea. There is so much additional complexity that would come form trying to optimize everything right away.

### References

Any accessed external documentation, tutorials and reference material will be declared in this section.

- [Unity Test Framework](https://github.com/ThrowTheSwitch/Unity/blob/master). MIT License. Code from their `src` directory was copied directly to `tests/framework` in this repository.
- [GGUF Format Specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md). Enums were copied from the pseudocode.
- [Gemma 4 Model on Hugging Face](https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF)
- [Hugging Face LLM Course (Tokenizers)](https://huggingface.co/learn/llm-course/en/chapter2/4)  
- [SentencePiece Tokenizer Documentation](https://github.com/google/sentencepiece)
- [A Visual Guide to Gemma 4](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4)
- [Float16 De-Quantization from libcanard](https://github.com/OpenCyphal/libcanard/blob/636795f4bc395f56af8d2c61d3757b5e762bb9e5/canard.c#L811-L834)
- [Google Gemini for some technical explanations](https://gemini.google.com/app)
- [PyTorch RMSNorm Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.normalization.RMSNorm.html)
- [PyTorch GELU Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html)
- [Mastering Gemma 4: A Comprehensive Deep Dive into Google's Next-Generation Open Model Architecture and Deployment](https://dev.to/jubinsoni/mastering-gemma-4-a-comprehensive-deep-dive-into-googles-next-generation-open-model-architecture-2f91)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) to dump known-good internal vectors to debug against.
- [Wikipedia: Attention (Machine Learning)](https://en.wikipedia.org/wiki/Attention_(machine_learning))
- [LLM Breakdown](https://mikexcohen.substack.com/p/llm-breakdown-46-transformer-outputs)
- [Feed Forward Network](https://sampathkumaran.medium.com/llms-simplified-feed-forward-network-ffn-24ec761e664a)
