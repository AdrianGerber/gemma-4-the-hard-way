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
#include <time.h>

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
static const GGUF_TensorInfo_t *GetTensorForBlock(Model_t *model, size_t blockIndex, const char *tensorName);
static float *ForwardProcess(Model_t *model, RuntimeData_t *runtimeData, uint32_t token, uint32_t position, bool preFill);
static uint32_t SelectTokenFromLogits(Model_t *model, float *logits);
static RuntimeData_t *AllocateRuntimeData(Model_t *model);
static void ReleaseRuntimeData(RuntimeData_t *data);
static void Dequantize(float *output, size_t outputSize, const GGUF_TensorInfo_t *input, size_t inputOffset);
static void RunAttention(Model_t *model, RuntimeData_t *runtimeData, uint32_t blockNumber, uint32_t position);
static void RunFeedForward(Model_t *model, RuntimeData_t *runtimeData, uint32_t blockNumber);
static float *RunClassifier(Model_t *model, RuntimeData_t *runtimeData);

static void AddFloats(float *out, const float *a, const float *b, size_t count);
static void MultiplyFloats(float *out, const float *a, const float *b, size_t count);
static void ScaleFloats(float *out, const float *in, float factor, size_t count);
static void CopyFloats(float *destination, const float *source, size_t count);
static void RMSNorm___old(float *output, const float *input, const float *weights, size_t count);
static void RMSNorm(float *output, const float *input, size_t count);
static void MultiplyMatrixAndVector(float *out, size_t outCount, const float *input, size_t inCount, const GGUF_TensorInfo_t *matrix);
static void ApplyRoPE(float *vector, size_t count, size_t headSize, uint32_t position, float frequencyBase);
static float DotProduct(float *a, float *b, size_t count);

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
        model->alignment = alignment->value.uint32;
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
    if (model->contextSize > 2048)
    {
        printf("Limiting context size to 2048 for now.\n");
        model->contextSize = 2048;
    }

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
    Tokenizer_DecodeToStdOut(model->tokenizer, (TokenizerEncoded_t){.length = 1, .tokens = &token}, false);
    position++;
    fflush(stdout);

    // Start predicting more future tokens by feeding the model's output back into itself.
    while (token != model->tokenizer.tokenIdEos)
    {
        logits = ForwardProcess(model, runtimeData, token, position, false);
        assert(logits);
        token = SelectTokenFromLogits(model, logits);
        Tokenizer_DecodeToStdOut(model->tokenizer, (TokenizerEncoded_t){.length = 1, .tokens = &token}, false);
        position++;

        fflush(stdout);

        if (position >= model->contextSize)
        {
            printf("\n Context size exceeded! Stopping.\n");
            break;
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

static const GGUF_TensorInfo_t *GetTensorForBlock(Model_t *model, size_t blockIndex, const char *tensorName)
{
    // e.g. blk.34.proj.weight
    char fullName[256];
    assert((int)sizeof(fullName) > snprintf(fullName, sizeof(fullName), "blk.%lu.%s", blockIndex, tensorName));
    return GGUF_TensorFindByName(model->tensorInfo, model->tensorInfoCount, fullName);
}

static float *ForwardProcess(Model_t *model, RuntimeData_t *runtimeData, uint32_t token, uint32_t position, bool preFill)
{
    /******************************************************************************
     * Token Embedding
     ******************************************************************************/
    // Embed the new token. The token is sent to the model as the initial state, but also
    // fed back in partially on each layer. For this there exist two different embeddings
    // that we need to load:

    // Main token embedding
    assert(model->weights.token_embd->dimensionCount == 2);
    assert(model->weights.token_embd->dimensions[0] == model->embeddingLength);
    assert(model->weights.token_embd->dimensions[1] > token);
    Dequantize(runtimeData->x, model->embeddingLength, model->weights.token_embd, token * model->embeddingLength);
    ScaleFloats(runtimeData->x, runtimeData->x, sqrtf((float)model->embeddingLength), model->embeddingLength);

    // Per-layer token embeddings. These are successively injected during the attention
    // block of each layer. This helps the model remember the original token while
    // information passes through the layers. We prepare all the data at this point,
    // but each layer will only inject 256 bytes at a time.
    assert(model->weights.per_layer_token_embd->dimensionCount == 2);
    assert(model->weights.per_layer_token_embd->dimensions[1] > token);
    const size_t injectedEmbeddingSizeTotal = model->weights.per_layer_token_embd->dimensions[0];
    assert(injectedEmbeddingSizeTotal % model->blockCount == 0);
    const size_t injectedEmbeddingSizePerLayer = injectedEmbeddingSizeTotal / model->blockCount;
    Dequantize(runtimeData->perLayerEmbeddings, model->weights.per_layer_token_embd->dimensions[0], model->weights.per_layer_token_embd, token * model->weights.per_layer_token_embd->dimensions[0]);
    ScaleFloats(runtimeData->perLayerEmbeddings, runtimeData->perLayerEmbeddings, sqrtf((float)injectedEmbeddingSizePerLayer), injectedEmbeddingSizeTotal);

    /******************************************************************************
     * Layer Processing
     ******************************************************************************/
    // Run the neural network layers
    for (size_t layerNumber = 0; layerNumber < model->blockCount; layerNumber++)
    {
        // <---------------------- Verified against llama.cpp ------------------>
        RunAttention(model, runtimeData, layerNumber, position);
        RunFeedForward(model, runtimeData, layerNumber);
    }

    // We can skip calculating the logits during the pre-fill phase.
    float *logits = NULL;
    if (!preFill)
    {
        logits = RunClassifier(model, runtimeData);
    }

    // Statistics to see just how slow the code runs :).
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec + end.tv_nsec / 1000000000.0) - (start.tv_sec + start.tv_nsec / 1000000000.0);
    printf("\nForward pass took %lfs.\n", elapsed);
    return logits;
}

static uint32_t SelectTokenFromLogits(Model_t *model, float *logits)
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

static RuntimeData_t *AllocateRuntimeData(Model_t *model)
{
    RuntimeData_t *tmp = malloc(sizeof(RuntimeData_t));
    assert(tmp);
    tmp->logits = malloc(model->tokenCount * sizeof(float));
    assert(tmp->logits);
    tmp->x = malloc(model->embeddingLength * sizeof(float));
    tmp->residuals = malloc(model->embeddingLength * sizeof(float));
    tmp->tmp1 = malloc(model->embeddingLength * sizeof(float));
    tmp->tmp2 = malloc(model->embeddingLength * sizeof(float));
    tmp->perLayerEmbeddings = malloc(model->weights.per_layer_token_embd->dimensions[0] * sizeof(float));
    tmp->downProjected = malloc(model->weights.per_layer_token_embd->dimensions[0] / model->blockCount * sizeof(float));
    assert(tmp->x);
    assert(tmp->residuals);
    assert(tmp->tmp1);
    assert(tmp->tmp2);
    assert(tmp->perLayerEmbeddings);
    assert(tmp->downProjected);

    // Allocate buffers for the feed forward network
    const GGUF_Metadata_t *feedForwardLengths = GGUF_MetadataFindByKey(model->metadata, model->metadataCount, "gemma4.feed_forward_length");
    assert(feedForwardLengths);
    assert(feedForwardLengths->type == GGUF_METADATA_VALUE_TYPE_ARRAY);
    assert(feedForwardLengths->value.array.type == GGUF_METADATA_VALUE_TYPE_INT32);
    int32_t largestForwardBuffer = 0;
    for (size_t i = 0; i < feedForwardLengths->value.array.length; i++)
    {
        const int32_t value = feedForwardLengths->value.array.array[i].int32;
        if (value > largestForwardBuffer)
            largestForwardBuffer = value;
    }
    tmp->ffnHiddenGate = malloc((size_t)largestForwardBuffer * sizeof(float));
    tmp->ffnHiddenUp = malloc((size_t)largestForwardBuffer * sizeof(float));
    assert(tmp->ffnHiddenGate);
    assert(tmp->ffnHiddenUp);

    tmp->attentionScores = malloc(model->contextSize * sizeof(float));
    assert(tmp->attentionScores);

    // Allocate buffers for the QKV attention calculations
    size_t maxQ = 0;
    size_t maxK = 0;
    size_t maxV = 0;
    tmp->kvCacheOffsets = malloc(model->blockCount * sizeof(size_t));
    assert(tmp->kvCacheOffsets);
    size_t kvCacheSize = 0;
    for (size_t block = 0; block < model->blockCount; block++)
    {
        tmp->kvCacheOffsets[block] = kvCacheSize;
        const GGUF_TensorInfo_t *q = GetTensorForBlock(model, block, "attn_q.weight");
        const GGUF_TensorInfo_t *k = GetTensorForBlock(model, block, "attn_k.weight");
        const GGUF_TensorInfo_t *v = GetTensorForBlock(model, block, "attn_v.weight");
        assert(q && k && v);
        assert(q->dimensionCount == 2 && k->dimensionCount == 2 && v->dimensionCount == 2);
        if (q->dimensions[1] > maxQ)
            maxQ = q->dimensions[1];
        if (k->dimensions[1] > maxK)
            maxK = k->dimensions[1];
        if (v->dimensions[1] > maxV)
            maxV = v->dimensions[1];

        assert(maxK == maxV);
        kvCacheSize += model->contextSize * (k->dimensions[1] + v->dimensions[1]);
    }

    tmp->q = malloc(maxQ * sizeof(float));
    tmp->k = malloc(maxK * sizeof(float));
    tmp->v = malloc(maxV * sizeof(float));
    assert(tmp->q);
    assert(tmp->k);
    assert(tmp->v);

    tmp->vMixed = malloc(maxQ * sizeof(float));
    assert(tmp->vMixed);

    // KV cache
    printf("Allocating %lu floats for KV cache\n", kvCacheSize);
    tmp->kvCache = malloc(kvCacheSize * sizeof(float));
    assert(tmp->kvCache);
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
        if (data->residuals)
            free(data->residuals);
        if (data->tmp1)
            free(data->tmp1);
        if (data->tmp2)
            free(data->tmp2);
        if (data->perLayerEmbeddings)
            free(data->perLayerEmbeddings);
        if (data->downProjected)
            free(data->downProjected);
        if (data->ffnHiddenGate)
            free(data->ffnHiddenGate);
        if (data->ffnHiddenUp)
            free(data->ffnHiddenUp);
        if (data->q)
            free(data->q);
        if (data->k)
            free(data->k);
        if (data->v)
            free(data->v);
        if (data->kvCache)
            free(data->kvCache);
        if (data->kvCacheOffsets)
            free(data->kvCacheOffsets);
        if (data->attentionScores)
            free(data->attentionScores);
        if (data->vMixed)
            free(data->vMixed);

        free(data);
    }
}

static void Dequantize(float *output, size_t outputSize, const GGUF_TensorInfo_t *input, size_t inputOffset)
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

static void RunAttention(Model_t *model, RuntimeData_t *runtimeData, uint32_t blockNumber, uint32_t position)
{
    const BlockWeights_t *layerWeights = model->weights.blocks + blockNumber;

    /******************************************************************************
     * Normalization
     ******************************************************************************/
    RMSNorm(runtimeData->tmp1, runtimeData->x, model->embeddingLength);

    assert(layerWeights->attn_norm->type == GGML_TYPE_F32);
    assert(layerWeights->attn_norm->dimensions[0] == model->embeddingLength);
    MultiplyFloats(runtimeData->tmp1, runtimeData->tmp1, layerWeights->attn_norm->data.float32, model->embeddingLength);

    /******************************************************************************
     * Compute Query, Key and Value Vectors
     ******************************************************************************/
    // General preparations
    assert(layerWeights->attn_q->dimensionCount == 2);
    assert(layerWeights->attn_k->dimensionCount == 2);
    assert(layerWeights->attn_v->dimensionCount == 2);
    assert(layerWeights->attn_k_norm->dimensionCount == 1);
    const size_t qDimension = layerWeights->attn_q->dimensions[1];
    const size_t kDimension = layerWeights->attn_k->dimensions[1];
    const size_t vDimension = layerWeights->attn_v->dimensions[1];
    const size_t headDimension = layerWeights->attn_k_norm->dimensions[0];
    const size_t headCount = qDimension / headDimension;
    assert(layerWeights->attn_q_norm->type == GGML_TYPE_F32);
    assert(layerWeights->attn_q_norm->dimensions[0] == headDimension);

    // Apply projection matrices
    MultiplyMatrixAndVector(runtimeData->q, qDimension, runtimeData->tmp1, model->embeddingLength, layerWeights->attn_q);
    MultiplyMatrixAndVector(runtimeData->k, kDimension, runtimeData->tmp1, model->embeddingLength, layerWeights->attn_k);
    MultiplyMatrixAndVector(runtimeData->v, vDimension, runtimeData->tmp1, model->embeddingLength, layerWeights->attn_v);

    // Normalization
    for (size_t head = 0; head < headCount; head++)
    {
        const size_t offset = head * headDimension;
        RMSNorm(runtimeData->q + offset, runtimeData->q + offset, headDimension);
        MultiplyFloats(runtimeData->q + offset, runtimeData->q + offset, layerWeights->attn_q_norm->data.float32, headDimension);
    }

    RMSNorm(runtimeData->k, runtimeData->k, kDimension);
    RMSNorm(runtimeData->v, runtimeData->v, vDimension);
    assert(layerWeights->attn_k_norm->type == GGML_TYPE_F32);
    assert(layerWeights->attn_k_norm->dimensions[0] == kDimension);
    MultiplyFloats(runtimeData->k, runtimeData->k, layerWeights->attn_k_norm->data.float32, kDimension);

    // RoPE
    const float ropeBase = (headDimension == 512) ? 1000000.0f : 10000.0f;
    ApplyRoPE(runtimeData->q, qDimension, headDimension, position, ropeBase);
    ApplyRoPE(runtimeData->k, kDimension, headDimension, position, ropeBase);

    // <---------------------- Verified against llama.cpp ------------------>

    // Apply RoPe.
    // TODO: calculate these constants!
    const bool globalAttention = (blockNumber % 5 == 4);
    const size_t headDimension = globalAttention ? 512 : 256;
    const size_t tokenKVSize = layerWeights->attn_k->dimensions[1] + layerWeights->attn_v->dimensions[1];
    const float ropeBase = globalAttention ? 1000000.0f : 10000.0f;
    ApplyRoPE(runtimeData->q, layerWeights->attn_q->dimensions[1], headDimension, position, ropeBase);
    ApplyRoPE(runtimeData->k, layerWeights->attn_k->dimensions[1], headDimension, position, ropeBase);

    // Save values to kv cache
    const size_t kSize = layerWeights->attn_k->dimensions[1];
    const size_t vSize = layerWeights->attn_v->dimensions[1];
    const size_t totalSize = kSize + vSize;
    const size_t cacheStartOffset = runtimeData->kvCacheOffsets[blockNumber];
    const size_t cachePositionOffset = position * totalSize;
    float *kvStart = runtimeData->kvCache + cacheStartOffset + cachePositionOffset;
    memcpy(kvStart, runtimeData->k, kSize * sizeof(float));
    memcpy(kvStart + kSize, runtimeData->v, vSize * sizeof(float));

    // Compute the attention
    const size_t qHeads = layerWeights->attn_q->dimensions[1] / headDimension;
    const size_t kvHeads = layerWeights->attn_k->dimensions[1] / headDimension;
    assert(qHeads % kvHeads == 0);
    const size_t qPerKv = qHeads / kvHeads;
    const float scale = 1.0f / sqrtf(headDimension);
    // Decide how far back to look
    const size_t startPos = (!globalAttention && position > 512) ? position - 512 : 0;
    // Process query heads
    for (size_t head = 0; head < qHeads; head++)
    {
        // Identify memory locations
        const size_t kvHeadIndex = head / qPerKv;
        float *qHead = runtimeData->q + (head * headDimension);
        float *outputHead = runtimeData->vMixed + (head * headDimension);

        // Quantify similarity
        for (size_t pos = startPos; pos <= position; pos++)
        {
            float *kHead = runtimeData->kvCache + runtimeData->kvCacheOffsets[blockNumber] + (pos * tokenKVSize) + (kvHeadIndex * headDimension);
            runtimeData->attentionScores[pos] = DotProduct(kHead, qHead, headDimension) * scale;
        }

        // Softmax
        float maxScore = -INFINITY;
        for (size_t pos = startPos; pos <= position; pos++)
        {
            if (runtimeData->attentionScores[pos] > maxScore)
                maxScore = runtimeData->attentionScores[pos];
        }
        float sum = 0.0f;
        for (size_t pos = startPos; pos <= position; pos++)
        {
            runtimeData->attentionScores[pos] = expf(runtimeData->attentionScores[pos] - maxScore);
            sum += runtimeData->attentionScores[pos];
        }
        for (size_t pos = startPos; pos <= position; pos++)
        {
            runtimeData->attentionScores[pos] /= sum;
        }

        // Mix the values
        memset(outputHead, 0, headDimension * sizeof(float));
        for (size_t pos = startPos; pos <= position; pos++)
        {
            float *vHead = runtimeData->kvCache + runtimeData->kvCacheOffsets[blockNumber] + (pos * tokenKVSize) + kSize + (kvHeadIndex * headDimension);
            for (size_t i = 0; i < headDimension; i++)
            {
                outputHead[i] += runtimeData->attentionScores[pos] * vHead[i];
            }
        }
    }

    // Trap NaNs inside the attention pool
    bool vMixedCorrupted = false;
    for (size_t i = 0; i < layerWeights->attn_q->dimensions[1]; i++)
    {
        if (isnan(runtimeData->vMixed[i]) || isinf(runtimeData->vMixed[i]))
        {
            printf("FATAL: vMixed contains NaN/Inf at index %zu!\n", i);
            vMixedCorrupted = true;
            break; // Stop at the first corrupted float
        }
    }

    if (!vMixedCorrupted)
    {
        printf("vMixed is 100%% clean. The math is exploding inside MultiplyMatrixAndVector!\n");
    }

    // Output projection
    assert(layerWeights->attn_output);
    MultiplyMatrixAndVector(runtimeData->tmp2, model->embeddingLength, runtimeData->vMixed, layerWeights->attn_q->dimensions[1], layerWeights->attn_output);
    printf("1. After attn_output proj: %f\n", runtimeData->tmp2[0]);

    // Post-normalization
    assert(layerWeights->post_attention_norm);
    assert(layerWeights->post_attention_norm->type == GGML_TYPE_F32);
    RMSNorm(runtimeData->tmp2, runtimeData->tmp2, layerWeights->post_attention_norm->data.float32, model->embeddingLength);
    printf("2. After post_attn_norm: %f\n", runtimeData->tmp2[0]);

    assert(layerWeights->layer_output_scale);
    assert(layerWeights->layer_output_scale->dimensionCount == 1);
    assert(layerWeights->layer_output_scale->dimensions[0] == 1);
    assert(layerWeights->layer_output_scale->type == GGML_TYPE_F32);
    ScaleFloats(runtimeData->tmp2, runtimeData->tmp2, layerWeights->layer_output_scale->data.float32[0], model->embeddingLength);
    printf("3. After layer scale: %f\n", runtimeData->tmp2[0]);

    // Add the residuals back in
    AddFloats(runtimeData->x, runtimeData->residuals, runtimeData->tmp2, model->embeddingLength);
    printf("4. After adding residuals: %f\n", runtimeData->x[0]);
}

static void RunFeedForward(Model_t *model, RuntimeData_t *runtimeData, uint32_t blockNumber)
{
    const BlockWeights_t *layerWeights = model->weights.blocks + blockNumber;

    // Save the residuals for later
    CopyFloats(runtimeData->residuals, runtimeData->x, model->embeddingLength);

    // Normalize the input vector before feeding it into the network
    assert(layerWeights->ffn_norm);
    assert(layerWeights->ffn_norm->type == GGML_TYPE_F32);
    RMSNorm(runtimeData->tmp1, runtimeData->x, layerWeights->ffn_norm->data.float32, model->embeddingLength);
    printf("5. After FFN norm: %f\n", runtimeData->tmp1[0]);

    // Expand to the hidden dimension, apply the activation function and then project back down.
    // The size differs by layer. Buffers ffnHiddenGate and ffnHiddenUp are allocated for the largest case.
    assert(layerWeights->ffn_gate->dimensionCount == 2);
    const size_t hiddenDimension = layerWeights->ffn_gate->dimensions[1];
    MultiplyMatrixAndVector(runtimeData->ffnHiddenGate, hiddenDimension, runtimeData->tmp1, model->embeddingLength, layerWeights->ffn_gate);
    MultiplyMatrixAndVector(runtimeData->ffnHiddenUp, hiddenDimension, runtimeData->tmp1, model->embeddingLength, layerWeights->ffn_up);
    for (size_t i = 0; i < hiddenDimension; i++)
    {
        const float x = runtimeData->ffnHiddenGate[i];
        const float gelu = x * 0.5 * (1.0f + erff(x / 1.41421356f));
        runtimeData->ffnHiddenGate[i] = gelu * runtimeData->ffnHiddenUp[i];
    }
    MultiplyMatrixAndVector(runtimeData->tmp2, model->embeddingLength, runtimeData->ffnHiddenGate, hiddenDimension, layerWeights->ffn_down);

    // Post-normalization after FFN
    assert(layerWeights->post_ffw_norm);
    assert(layerWeights->post_ffw_norm->type == GGML_TYPE_F32);
    RMSNorm(runtimeData->tmp2, runtimeData->tmp2, layerWeights->post_ffw_norm->data.float32, model->embeddingLength);

    assert(layerWeights->layer_output_scale);
    assert(layerWeights->layer_output_scale->dimensionCount == 1);
    assert(layerWeights->layer_output_scale->dimensions[0] == 1);
    assert(layerWeights->layer_output_scale->type == GGML_TYPE_F32);
    ScaleFloats(runtimeData->tmp2, runtimeData->tmp2, layerWeights->layer_output_scale->data.float32[0], model->embeddingLength);

    // Add the residuals back in
    AddFloats(runtimeData->x, runtimeData->residuals, runtimeData->tmp2, model->embeddingLength);

    // Post-normalization on the final combined state
    // assert(layerWeights->post_norm);
    // assert(layerWeights->post_norm->type == GGML_TYPE_F32);
    // RMSNorm(runtimeData->x, runtimeData->x, layerWeights->post_norm->data.float32, model->embeddingLength);
}

static float *RunClassifier(Model_t *model, RuntimeData_t *runtimeData)
{
    // Normalization
    assert(model->weights.output_norm);
    assert(model->weights.output_norm->type == GGML_TYPE_F32);
    RMSNorm(runtimeData->tmp1, runtimeData->x, model->weights.output_norm->data.float32, model->embeddingLength);

    // Prepare the LM head tensor (same as token_embd due to 'weight tying')
    const GGUF_TensorInfo_t *head = model->weights.token_embd;
    assert(head->dimensionCount == 2);
    assert(model->tokenCount == head->dimensions[1]);
    MultiplyMatrixAndVector(runtimeData->logits, head->dimensions[1], runtimeData->tmp1, model->embeddingLength, head);
    return runtimeData->logits;
}

static void AddFloats(float *out, const float *a, const float *b, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = a[i] + b[i];
    }
}
static void MultiplyFloats(float *out, const float *a, const float *b, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = a[i] * b[i];
    }
}

static void ScaleFloats(float *out, const float *in, float factor, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = factor * in[i];
    }
}

static void CopyFloats(float *destination, const float *source, size_t count)
{
    memcpy(destination, source, count * sizeof(float));
}

static void RMSNorm___old(float *output, const float *input, const float *weights, size_t count)
{
    double sum = 0.0;
    for (size_t i = 0; i < count; i++)
    {
        sum += (double)input[i] * input[i];
    }

    float normalizingFactor = 1.0f / sqrtf((float)(sum / count) + 1e-6f);
    if (weights)
    {
    for (size_t i = 0; i < count; i++)
    {
        // Scale by the computed factor while also introducing the learned weights.
        // From a few web searches, it looks like pure normalization would be too
        // restrictive for the neural network. Adding an additional step multiplying by
        // the learned weights allows important features to be highlighted / preserved better.
            output[i] = (input[i] * normalizingFactor) * (weights[i]);
        }
    }
    else
    {
        for (size_t i = 0; i < count; i++)
        {
            output[i] = input[i] * normalizingFactor;
        }
    }
}

static void RMSNorm(float *output, const float *input, size_t count)
{
    double sum = 0.0;
    for (size_t i = 0; i < count; i++)
    {
        sum += (double)input[i] * input[i];
    }
    float normalizingFactor = 1.0f / sqrtf((float)(sum / count) + 1e-6f);

    for (size_t i = 0; i < count; i++)
    {
        output[i] = (input[i] * normalizingFactor);
    }
}

static void MultiplyMatrixAndVector(float *out, size_t outCount, const float *input, size_t inCount, const GGUF_TensorInfo_t *matrix)
{
    assert(matrix->dimensionCount == 2);
    const size_t rows = matrix->dimensions[1];
    const size_t cols = matrix->dimensions[0];

    assert(rows == outCount);
    assert(cols == inCount);

    if (matrix->type == GGML_TYPE_Q8_0)
    {
        // Only support sizes that are a multiple of quantization block size for now.
        const size_t blockSize = sizeof(matrix->data.q8_0->quantized) / sizeof(matrix->data.q8_0->quantized[0]);
        assert(cols % blockSize == 0);
        const size_t blocksPerRow = cols / blockSize;

        // Calculate each row of the output vector as the dot product between input vector
        // and the column of the matrix.
        for (size_t row = 0; row < rows; row++)
        {
            float dotProduct = 0.0f;
            const GGUF_Q8_0_t *rowBlocks = matrix->data.q8_0 + blocksPerRow * row;
            for (size_t block = 0; block < blocksPerRow; block++)
            {
                const float scale = GGUF_Float16ToFloat(rowBlocks[block].scale);
                for (size_t quantizedIndex = 0; quantizedIndex < blockSize; quantizedIndex++)
                {
                    const float matrixWeight = scale * (float)rowBlocks[block].quantized[quantizedIndex];
                    dotProduct += input[block * blockSize + quantizedIndex] * matrixWeight;
                }
            }
            out[row] = dotProduct;
        }
    }
    else
    {
        assert(false);
    }
}

static void ApplyRoPE(float *vector, size_t count, size_t headSize, uint32_t position, float frequencyBase)
{
    assert(count % headSize == 0);
    size_t numberOfHeads = count / headSize;

    for (size_t headIndex = 0; headIndex < numberOfHeads; headIndex++)
    {
        float *head = vector + headIndex * headSize;

        // Rotate pairs of x/y
        for (size_t i = 0; i < headSize; i += 2)
        {
            const float exponent = (float)i / headSize;
            const float inverseFrequency = 1.0f / powf(frequencyBase, exponent);
            const float theta = (float)position * inverseFrequency;

            const float cosTheta = cosf(theta);
            const float sinTheta = sinf(theta);
            const float x = head[i];
            const float y = head[i + 1];

            head[i] = x * cosTheta - y * sinTheta;
            head[i + 1] = x * sinTheta + y * cosTheta;
        }
    }
}

static float DotProduct(float *a, float *b, size_t count)
{
    float sum = 0.0f;
    for (size_t i = 0; i < count; i++)
    {
        sum += a[i] * b[i];
    }
    return sum;
}
