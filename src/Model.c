/**
 * @file      Model.c
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

/******************************************************************************
 * Includes
 ******************************************************************************/
#include "Model.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

/******************************************************************************
 * Private Constants and Macros
 ******************************************************************************/

/******************************************************************************
 * Private Type Definitions
 ******************************************************************************/

/******************************************************************************
 * Private / Static Variables
 ******************************************************************************/

/******************************************************************************
 * Private Function Prototypes
 ******************************************************************************/

/******************************************************************************
 * Public Function Implementations
 ******************************************************************************/

Model_t *Model_LoadFromGGUF(const uint8_t *data, size_t length)
{
    if (!GGUF_VerifyHeader(data, length))
    {
        fprintf(stderr, "Cannot decode GGUF file.\n");
        return NULL;
    }

    // Parse metadata
    Model_t *model = malloc(sizeof(Model_t));
    assert(model);
    model->metadataCount = GGUF_GetMetadataCount(data);
    model->metadata = malloc(model->metadataCount * sizeof(GGUF_Metadata_t));
    assert(model->metadata);

    const uint8_t *readPointer = GGUF_SkipHeader(data);
    for (size_t i = 0; i < model->metadataCount; i++)
    {
        model->metadata[i] = GGUF_MetadataFromMemory(&readPointer);

        if (model->metadata[i].type == GGUF_METADATA_VALUE_TYPE_ARRAY && model->metadata[i].value.array.length > 10)
        {
            printf("'%s' = [...]\n", model->metadata[i].key);
        }
        else if (model->metadata[i].type == GGUF_METADATA_VALUE_TYPE_STRING && strlen(model->metadata[i].value.str) > 1024)
        {
            printf("'%s' = '...'\n", model->metadata[i].key);
        }
        else
        {
            GGUF_MetadataPrint(model->metadata[i]);
        }
    }

    // Parse tensor infos
    printf("\nTensors:\n");
    model->tensorInfoCount = GGUF_GetTensorCount(data);
    model->tensorInfo = malloc(model->tensorInfoCount * sizeof(GGUF_TensorInfo_t));
    assert(model->tensorInfo);
    for (size_t i = 0; i < model->tensorInfoCount; i++)
    {
        model->tensorInfo[i] = GGUF_TensorInfoFromMemory(&readPointer, data);
        GGUF_TensorInfoPrint(model->tensorInfo[i]);
    }

    model->tokenizer = Tokenizer_Init(model->metadata, model->metadataCount);

    return model;
}

void Model_GenerateCompletionsToStdOut(Model_t *model, const char *prompt)
{
    printf("Encoding Text '%s'\n", prompt);
    TokenizerEncoded_t tokenIds = Tokenizer_Encode(model->tokenizer, prompt);
    printf("Tokenized: ");
    Tokenizer_DecodeToStdOut(model->tokenizer, tokenIds, true);
    printf("\n");
}

void Model_Release(Model_t *model)
{
    if (model->metadata)
    {
        for (size_t i = 0; i < model->metadataCount; i++)
        {
            GGUF_MetadataRelease(model->metadata[i]);
        }
        free(model->metadata);
    }

    if (model->tensorInfo)
    {
        for (size_t i = 0; i < model->tensorInfoCount; i++)
        {
            GGUF_TensorInfoRelease(model->tensorInfo[i]);
        }
        free(model->tensorInfo);
    }
    Tokenizer_Release(model->tokenizer);

    free(model);
}

/******************************************************************************
 * Private Function Implementations
 ******************************************************************************/
