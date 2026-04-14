/**
 * @file      main.c
 * @author    Adrian Gerber
 * @brief     C main function implementation.
 * @copyright Copyright (c) 2026 Adrian Gerber
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "GGUF.h"
#include <string.h>
#include <assert.h>
#include "Model.h"

int main(void)
{
    // Map the file content into memory
    const char *filename = "gemma-4-e2b-it-Q8_0.gguf";
    int f = open(filename, O_RDONLY);
    if (f == -1)
    {
        perror("open failed");
        exit(1);
    }
    struct stat fileStat;
    if (fstat(f, &fileStat) != 0)
    {
        perror("stat failed");
        close(f);
        exit(1);
    }
    const uint8_t *data = mmap(NULL, (size_t)fileStat.st_size, PROT_READ, MAP_SHARED, f, 0);
    if (data == MAP_FAILED)
    {
        perror("mmap failed");
        close(f);
        exit(1);
    }

    // Initialize the model and run completions.
    Model_t *model = Model_LoadFromGGUF(data, (size_t)fileStat.st_size);
    if (model)
    {
        printf("\n\n");

        const char *prompt =
            "<bos>\n"
            "<|turn>user\n"
            "Hello world<turn|>\n"
            "<|turn>model\n";

        Model_GenerateCompletionsToStdOut(model, prompt);

        Model_Release(model);
        model = NULL;
    }
    else
    {
        fprintf(stderr, "Failed to load model.\n");
    }

    // Cleanup
    munmap((void *)data, (size_t)fileStat.st_size);
    close(f);
    return 0;
}
