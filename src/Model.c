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
const GGUF_TensorInfo_t *GetTensorForBlock(Model_t *model, size_t blockIndex, const char *tensorName);

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
    model->tensorInfoCount = GGUF_GetTensorCount(data);
    model->tensorInfo = malloc(model->tensorInfoCount * sizeof(GGUF_TensorInfo_t));
    assert(model->tensorInfo);
    for (size_t i = 0; i < model->tensorInfoCount; i++)
    {
        model->tensorInfo[i] = GGUF_TensorInfoFromMemory(&readPointer, data);
        // GGUF_TensorInfoPrint(model->tensorInfo[i]);
    }

    // Tokenizer
    printf("\n");
    model->tokenizer = Tokenizer_Init(model->metadata, model->metadataCount);

    // Load blocks
    const GGUF_Metadata_t *blockCount = GGUF_MetadataFindByKey(model->metadata, model->metadataCount, "gemma4.block_count");
    assert(blockCount);
    assert(blockCount->type == GGUF_METADATA_VALUE_TYPE_UINT32);
    model->blockCount = blockCount->value.uint32;
    model->blocks = malloc(model->blockCount * sizeof(Block_t));
    printf("Loading %lu blocks...\n", model->blockCount);
    for (size_t i = 0; i < model->blockCount; i++)
    {
        model->blocks[i].attn_k = GetTensorForBlock(model, i, "attn_k.weight");
        model->blocks[i].attn_k_norm = GetTensorForBlock(model, i, "attn_k_norm.weight");
        model->blocks[i].attn_norm = GetTensorForBlock(model, i, "attn_norm.weight");
        model->blocks[i].attn_output = GetTensorForBlock(model, i, "attn_output.weight");
        model->blocks[i].attn_q = GetTensorForBlock(model, i, "attn_q.weight");
        model->blocks[i].attn_q_norm = GetTensorForBlock(model, i, "attn_q_norm.weight");
        model->blocks[i].attn_v = GetTensorForBlock(model, i, "attn_v.weight");
        model->blocks[i].ffn_down = GetTensorForBlock(model, i, "ffn_down.weight");
        model->blocks[i].ffn_gate = GetTensorForBlock(model, i, "ffn_gate.weight");
        model->blocks[i].ffn_norm = GetTensorForBlock(model, i, "ffn_norm.weight");
        model->blocks[i].ffn_up = GetTensorForBlock(model, i, "ffn_up.weight");
        model->blocks[i].inp_gate = GetTensorForBlock(model, i, "inp_gate.weight");
        model->blocks[i].layer_output_scale = GetTensorForBlock(model, i, "layer_output_scale.weight");
        model->blocks[i].post_attention_norm = GetTensorForBlock(model, i, "post_attention_norm.weight");
        model->blocks[i].post_ffw_norm = GetTensorForBlock(model, i, "post_ffw_norm.weight");
        model->blocks[i].post_norm = GetTensorForBlock(model, i, "post_norm.weight");
        model->blocks[i].proj = GetTensorForBlock(model, i, "proj.weight");
        // At least one tensor is needed per block
        assert(0 < (model->blocks[i].attn_k != NULL) + (model->blocks[i].attn_k_norm != NULL) + (model->blocks[i].attn_norm != NULL) + (model->blocks[i].attn_output != NULL) + (model->blocks[i].attn_q != NULL) + (model->blocks[i].attn_q_norm != NULL) + (model->blocks[i].attn_v != NULL) + (model->blocks[i].ffn_down != NULL) + (model->blocks[i].ffn_gate != NULL) + (model->blocks[i].ffn_norm != NULL) + (model->blocks[i].ffn_up != NULL) + (model->blocks[i].inp_gate != NULL) + (model->blocks[i].layer_output_scale != NULL) + (model->blocks[i].post_attention_norm != NULL) + (model->blocks[i].post_ffw_norm != NULL) + (model->blocks[i].post_norm != NULL) + (model->blocks[i].proj != NULL));
    }

    model->output_norm = GGUF_TensorFindByName(model->tensorInfo, model->tensorInfoCount, "output_norm.weight");
    model->per_layer_model_proj = GGUF_TensorFindByName(model->tensorInfo, model->tensorInfoCount, "per_layer_model_proj.weight");
    model->per_layer_proj_norm = GGUF_TensorFindByName(model->tensorInfo, model->tensorInfoCount, "per_layer_proj_norm.weight");
    model->per_layer_token_embd = GGUF_TensorFindByName(model->tensorInfo, model->tensorInfoCount, "per_layer_token_embd.weight");
    model->rope_freqs = GGUF_TensorFindByName(model->tensorInfo, model->tensorInfoCount, "rope_freqs.weight");
    model->token_embd = GGUF_TensorFindByName(model->tensorInfo, model->tensorInfoCount, "token_embd.weight");
    assert(model->output_norm);
    assert(model->per_layer_model_proj);
    assert(model->per_layer_proj_norm);
    assert(model->per_layer_token_embd);
    assert(model->rope_freqs);
    assert(model->token_embd);

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

    if (model->blocks)
    {
        free(model->blocks);
    }

    free(model);
}

/******************************************************************************
 * Private Function Implementations
 ******************************************************************************/

const GGUF_TensorInfo_t *GetTensorForBlock(Model_t *model, size_t blockIndex, const char *tensorName)
{
    // e.g. blk.34.proj.weight
    char fullName[256];
    assert((int)sizeof(fullName) > snprintf(fullName, sizeof(fullName), "blk.%lu.%s", blockIndex, tensorName));
    return GGUF_TensorFindByName(model->tensorInfo, model->tensorInfoCount, fullName);
}
