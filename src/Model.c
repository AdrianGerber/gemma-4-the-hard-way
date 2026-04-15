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
#include <math.h>

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
float *ForwardProcess(Model_t *model, RuntimeData_t *runtimeData, uint32_t token, uint32_t position, bool preFill);
uint32_t SelectTokenFromLogits(Model_t *model, float *logits);
RuntimeData_t *AllocateRuntimeData(Model_t *model);
void ReleaseRuntimeData(RuntimeData_t *data);
void Dequantize(float *output, size_t outputSize, const GGUF_TensorInfo_t *input, size_t inputOffset);

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
        model->tensorInfo[i] = GGUF_TensorInfoFromMemory(&readPointer);
        // GGUF_TensorInfoPrint(model->tensorInfo[i]);
    }

    // Handle padding between tensor_info and tensor_data sections
    model->alignment = 32;
    const GGUF_Metadata_t *alignment = GGUF_MetadataFindByKey(model->metadata, model->metadataCount, "general.alignment");
    if (alignment)
    {
        assert(alignment->type == GGUF_METADATA_VALUE_TYPE_UINT32);
        model->embeddingLength = alignment->value.uint32;
    }
    size_t currentFileOffset = ((size_t)(readPointer - data));
    size_t padding = (model->alignment - (currentFileOffset % model->alignment)) % model->alignment;
    const uint8_t *tensorData = readPointer + padding;

    // Get tensor weights
    for (size_t i = 0; i < model->tensorInfoCount; i++)
    {
        GGUF_TensorGetWeightsFromDataSection(model->tensorInfo + i, &tensorData);
    }

    // Tokenizer
    printf("\n");
    model->tokenizer = Tokenizer_Init(model->metadata, model->metadataCount);
    model->tokenCount = model->tokenizer.tokens.length;

    // Load blocks
    const GGUF_Metadata_t *blockCount = GGUF_MetadataFindByKey(model->metadata, model->metadataCount, "gemma4.block_count");
    assert(blockCount);
    assert(blockCount->type == GGUF_METADATA_VALUE_TYPE_UINT32);
    model->blockCount = blockCount->value.uint32;
    model->weights.blocks = malloc(model->blockCount * sizeof(BlockWeights_t));
    printf("Loading %lu blocks...\n", model->blockCount);
    for (size_t i = 0; i < model->blockCount; i++)
    {
        model->weights.blocks[i].attn_k = GetTensorForBlock(model, i, "attn_k.weight");
        model->weights.blocks[i].attn_k_norm = GetTensorForBlock(model, i, "attn_k_norm.weight");
        model->weights.blocks[i].attn_norm = GetTensorForBlock(model, i, "attn_norm.weight");
        model->weights.blocks[i].attn_output = GetTensorForBlock(model, i, "attn_output.weight");
        model->weights.blocks[i].attn_q = GetTensorForBlock(model, i, "attn_q.weight");
        model->weights.blocks[i].attn_q_norm = GetTensorForBlock(model, i, "attn_q_norm.weight");
        model->weights.blocks[i].attn_v = GetTensorForBlock(model, i, "attn_v.weight");
        model->weights.blocks[i].ffn_down = GetTensorForBlock(model, i, "ffn_down.weight");
        model->weights.blocks[i].ffn_gate = GetTensorForBlock(model, i, "ffn_gate.weight");
        model->weights.blocks[i].ffn_norm = GetTensorForBlock(model, i, "ffn_norm.weight");
        model->weights.blocks[i].ffn_up = GetTensorForBlock(model, i, "ffn_up.weight");
        model->weights.blocks[i].inp_gate = GetTensorForBlock(model, i, "inp_gate.weight");
        model->weights.blocks[i].layer_output_scale = GetTensorForBlock(model, i, "layer_output_scale.weight");
        model->weights.blocks[i].post_attention_norm = GetTensorForBlock(model, i, "post_attention_norm.weight");
        model->weights.blocks[i].post_ffw_norm = GetTensorForBlock(model, i, "post_ffw_norm.weight");
        model->weights.blocks[i].post_norm = GetTensorForBlock(model, i, "post_norm.weight");
        model->weights.blocks[i].proj = GetTensorForBlock(model, i, "proj.weight");
        // At least one tensor is needed per block
        assert(0 < (model->weights.blocks[i].attn_k != NULL) + (model->weights.blocks[i].attn_k_norm != NULL) + (model->weights.blocks[i].attn_norm != NULL) + (model->weights.blocks[i].attn_output != NULL) + (model->weights.blocks[i].attn_q != NULL) + (model->weights.blocks[i].attn_q_norm != NULL) + (model->weights.blocks[i].attn_v != NULL) + (model->weights.blocks[i].ffn_down != NULL) + (model->weights.blocks[i].ffn_gate != NULL) + (model->weights.blocks[i].ffn_norm != NULL) + (model->weights.blocks[i].ffn_up != NULL) + (model->weights.blocks[i].inp_gate != NULL) + (model->weights.blocks[i].layer_output_scale != NULL) + (model->weights.blocks[i].post_attention_norm != NULL) + (model->weights.blocks[i].post_ffw_norm != NULL) + (model->weights.blocks[i].post_norm != NULL) + (model->weights.blocks[i].proj != NULL));
    }

    model->weights.output_norm = GGUF_TensorFindByName(model->tensorInfo, model->tensorInfoCount, "output_norm.weight");
    model->weights.per_layer_model_proj = GGUF_TensorFindByName(model->tensorInfo, model->tensorInfoCount, "per_layer_model_proj.weight");
    model->weights.per_layer_proj_norm = GGUF_TensorFindByName(model->tensorInfo, model->tensorInfoCount, "per_layer_proj_norm.weight");
    model->weights.per_layer_token_embd = GGUF_TensorFindByName(model->tensorInfo, model->tensorInfoCount, "per_layer_token_embd.weight");
    model->weights.rope_freqs = GGUF_TensorFindByName(model->tensorInfo, model->tensorInfoCount, "rope_freqs.weight");
    model->weights.token_embd = GGUF_TensorFindByName(model->tensorInfo, model->tensorInfoCount, "token_embd.weight");
    assert(model->weights.output_norm);
    assert(model->weights.per_layer_model_proj);
    assert(model->weights.per_layer_proj_norm);
    assert(model->weights.per_layer_token_embd);
    assert(model->weights.rope_freqs);
    assert(model->weights.token_embd);

    const GGUF_Metadata_t *contextSize = GGUF_MetadataFindByKey(model->metadata, model->metadataCount, "gemma4.context_length");
    assert(contextSize);
    assert(contextSize->type == GGUF_METADATA_VALUE_TYPE_UINT32);
    model->contextSize = contextSize->value.uint32;
    const GGUF_Metadata_t *embeddingLength = GGUF_MetadataFindByKey(model->metadata, model->metadataCount, "gemma4.embedding_length");
    assert(embeddingLength);
    assert(embeddingLength->type == GGUF_METADATA_VALUE_TYPE_UINT32);
    model->embeddingLength = embeddingLength->value.uint32;
    return model;
}

void Model_GenerateCompletionsToStdOut(Model_t *model, const char *prompt)
{
    printf("Encoding Text '%s'\n", prompt);
    TokenizerEncoded_t tokenIds = Tokenizer_Encode(model->tokenizer, prompt);
    printf("Tokenized: ");
    Tokenizer_DecodeToStdOut(model->tokenizer, tokenIds, true);
    printf("\n");

    // Prefill phase to build the KV cache.
    printf("Prefilling KV Cache...\n");
    RuntimeData_t *runtimeData = AllocateRuntimeData(model);
    assert(runtimeData);

    uint32_t position = 0;
    for (size_t inputTokenIndex = 0; inputTokenIndex < tokenIds.length - 1; inputTokenIndex++)
    {
        ForwardProcess(model, runtimeData, tokenIds.tokens[inputTokenIndex], position++, true);
    }

    // Generate the first new token from the last input token.
    printf("Generating Predictions:\n");
    float *logits = ForwardProcess(model, runtimeData, tokenIds.tokens[position], position, false);
    assert(logits);
    uint32_t token = SelectTokenFromLogits(model, logits);
    Tokenizer_DecodeToStdOut(model->tokenizer, (TokenizerEncoded_t){.length = 1, .tokens = &token}, true);
    position++;

    // Start predicting more future tokens by feeding the model's output back into itself.
    while (token != model->tokenizer.tokenIdEos)
    {
        logits = ForwardProcess(model, runtimeData, token, position, false);
        assert(logits);
        token = SelectTokenFromLogits(model, logits);
        Tokenizer_DecodeToStdOut(model->tokenizer, (TokenizerEncoded_t){.length = 1, .tokens = &token}, true);
        position++;

        if (position >= model->contextSize)
        {
            printf("\n Context size exceeded! Stopping.\n");
        }
    }
    printf("\n");

    ReleaseRuntimeData(runtimeData);
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

    if (model->weights.blocks)
    {
        free(model->weights.blocks);
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

float *ForwardProcess(Model_t *model, RuntimeData_t *runtimeData, uint32_t token, uint32_t position, bool preFill)
{
    // Get the token's embedding
    assert(model->weights.token_embd->dimensionCount == 2);
    assert(model->weights.token_embd->dimensions[0] == model->embeddingLength);
    assert(model->weights.token_embd->dimensions[1] > token);
    Dequantize(runtimeData->x, model->embeddingLength, model->weights.token_embd, token * model->embeddingLength);

    (void)model;
    (void)runtimeData;
    (void)token;
    (void)position;

    // We can skip calculating the logits during the pre-fill phase.
    float *logits = NULL;
    if (!preFill)
    {
        logits = runtimeData->logits;
        logits[model->tokenizer.tokenIdEos] = 1.0f;
    }

    return logits;
}

uint32_t SelectTokenFromLogits(Model_t *model, float *logits)
{
    // TODO: Implement proper scaling and temperature handling.

    // For now, just pick the token with the highest logit.
    float highestLogitValue = -INFINITY;
    uint32_t highestLogitToken = 0;
    for (size_t i = 0; i < model->tokenCount; i++)
    {
        if (logits[i] > highestLogitValue)
        {
            highestLogitValue = logits[i];
            highestLogitToken = i;
        }
    }
    return highestLogitToken;
}

RuntimeData_t *AllocateRuntimeData(Model_t *model)
{
    RuntimeData_t *tmp = malloc(sizeof(RuntimeData_t));
    assert(tmp);
    tmp->logits = malloc(model->tokenCount * sizeof(float));
    tmp->x = malloc(model->embeddingLength * sizeof(float));
    return tmp;
}

void ReleaseRuntimeData(RuntimeData_t *data)
{
    if (data)
    {
        if (data->logits)
            free(data->logits);
        if (data->x)
            free(data->x);
        free(data);
    }
}

void Dequantize(float *output, size_t outputSize, const GGUF_TensorInfo_t *input, size_t inputOffset)
{
    switch (input->type)
    {
    case GGML_TYPE_Q8_0:
    {
        assert(sizeof(GGUF_Q8_0_t) == 34);

        // Time wasted here: 4h :)
        // Let's just say it took a while until I figured out that in the GGUF format convention, array dimensions are orderd
        // exactly opposite from what I expected (unlike in C, the fastest moving dimension is specified first). On top of that,
        // I also initially messed up the tensor offset and alignment in the GGUF file.

        // Only full blocks supported for now
        const size_t blockSize = sizeof(input->data.q8_0->quantized) / sizeof(input->data.q8_0->quantized[0]);
        assert((outputSize % blockSize) == 0);
        assert((inputOffset % blockSize) == 0);
        inputOffset /= blockSize;

        const size_t blockCount = outputSize / blockSize;
        size_t outputIndex = 0;
        for (size_t blockIndex = inputOffset; blockIndex < inputOffset + blockCount; blockIndex++)
        {
            const GGUF_Q8_0_t *block = input->data.q8_0 + blockIndex;
            const float scale = GGUF_Float16ToFloat(block->scale);
            for (size_t quantizedIndex = 0; quantizedIndex < blockSize; quantizedIndex++)
            {
                output[outputIndex++] = scale * ((float)block->quantized[quantizedIndex]);
            }
        }
        break;
    }
    default:
        assert(false);
    }
}
