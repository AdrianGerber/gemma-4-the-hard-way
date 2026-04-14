/**
 * @file      Model.h
 * @author    Adrian Gerber
 * @brief     Bare-minimum implementation for working with language models.
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

#ifndef MODEL_H_
#define MODEL_H_

/******************************************************************************
 * Includes
 ******************************************************************************/
#include <stdint.h>
#include <stdbool.h>
#include "GGUF.h"
#include "Tokenizer.h"

/******************************************************************************
 * Constants and Macros
 ******************************************************************************/

/******************************************************************************
 * Type Definitions
 ******************************************************************************/

/**
 * @brief Structure representing the information needed to run a language model.
 *
 */
typedef struct
{
    GGUF_Metadata_t *metadata;
    GGUF_TensorInfo_t *tensorInfo;
    size_t metadataCount, tensorInfoCount;
    Tokenizer_t tokenizer;
} Model_t;

/******************************************************************************
 * Public Function Prototypes
 ******************************************************************************/

/**
 * @brief Initialize the language model from raw bytes of a *.gguf file.
 *
 * @param data Raw bytes in memory.
 * @param length Number of bytes.
 * @return Model_t* Resulting model instance. NULL on error.
 */
Model_t *Model_LoadFromGGUF(const uint8_t *data, size_t length);

/**
 * @brief Run inference on the language model and print the results to stdout.
 *
 * @param model Instance.
 * @param prompt Initial text input.
 */
void Model_GenerateCompletionsToStdOut(Model_t *model, const char *prompt);

/**
 * @brief Clean up the memory allocated for a mode.
 *
 * @param model Instance pointer.
 */
void Model_Release(Model_t *model);

#endif /* MODEL_H_ */