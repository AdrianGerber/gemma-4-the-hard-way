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

#define DEBUG_PRINT 0

#if DEBUG_PRINT
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
// Ugly but very, very useful debug macro :)
#define DEBUG_VECTOR(tag, vector, length)                                                                                                                                                       \
    {                                                                                                                                                                                           \
        float tmpSum123 = 0.0f;                                                                                                                                                                 \
        for (size_t i = 0; i < length; i++)                                                                                                                                                     \
            tmpSum123 += vector[i];                                                                                                                                                             \
        printf("%04u %-20s %-30s sum=%06f            value=[%f, %f, ..., %f, %f]\n", __LINE__, tag, TOSTRING(vector), tmpSum123, vector[0], vector[1], vector[length - 2], vector[length - 1]); \
    }
#else

#define DEBUG_VECTOR(tag, vector, length)

#endif

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
static void RunInjection(Model_t *model, RuntimeData_t *runtimeData, uint32_t blockNumber);
static float *RunClassifier(Model_t *model, RuntimeData_t *runtimeData);

static void AddFloats(float *out, const float *a, const float *b, size_t count);
static void MultiplyFloats(float *out, const float *a, const float *b, size_t count);
static void ScaleFloats(float *out, const float *in, float factor, size_t count);
static void CopyFloats(float *destination, const float *source, size_t count);
static void AddFloatsScaled(float *out, const float *in, const float *added, float scale, size_t count);
static void RMSNorm(float *output, const float *input, size_t count);
static void MultiplyMatrixAndVector(float *out, size_t outCount, const float *input, size_t inCount, const GGUF_TensorInfo_t *matrix);
static void ApplyRoPE(float *vector, size_t count, size_t headSize, uint32_t position, float frequencyBase);
static float DotProduct(const float *a, const float *b, size_t count);

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
    printf("Generating Predictions:\n\n");
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
    DEBUG_VECTOR("", runtimeData->x, model->embeddingLength);

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
    DEBUG_VECTOR("", runtimeData->perLayerEmbeddings, injectedEmbeddingSizeTotal);

    MultiplyMatrixAndVector(runtimeData->allLayerModelProjections, injectedEmbeddingSizeTotal, runtimeData->x, model->embeddingLength, model->weights.per_layer_model_proj);
    ScaleFloats(runtimeData->allLayerModelProjections, runtimeData->allLayerModelProjections, 1.0f / sqrtf(model->embeddingLength), injectedEmbeddingSizeTotal);
    DEBUG_VECTOR("", runtimeData->allLayerModelProjections, injectedEmbeddingSizeTotal);

    /******************************************************************************
     * Layer Processing
     ******************************************************************************/
    // Run the neural network layers
    for (size_t layerNumber = 0; layerNumber < model->blockCount; layerNumber++)
    {
#if DEBUG_PRINT
        printf("\n--- Layer %lu ---\n", layerNumber);
#endif
        RunAttention(model, runtimeData, layerNumber, position);
        RunFeedForward(model, runtimeData, layerNumber);
        RunInjection(model, runtimeData, layerNumber);
    }

    // Final logit calculation
    float *logits = NULL;
    if (!preFill)
    {
        logits = RunClassifier(model, runtimeData);
    }

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
    assert(tmp->x);
    assert(tmp->residuals);
    assert(tmp->tmp1);
    assert(tmp->tmp2);

    assert(model->weights.per_layer_token_embd);
    const size_t totalInjectedSize = model->weights.per_layer_token_embd->dimensions[0];
    const size_t perLayerSize = totalInjectedSize / model->blockCount;
    tmp->perLayerEmbeddings = malloc(totalInjectedSize * sizeof(float));
    tmp->allLayerModelProjections = malloc(totalInjectedSize * sizeof(float));
    tmp->downProjected = malloc(perLayerSize * sizeof(float));
    assert(tmp->perLayerEmbeddings);
    assert(tmp->allLayerModelProjections);
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
    tmp->kvCache = malloc(kvCacheSize * sizeof(float));
    tmp->attentionScores = malloc(model->contextSize * sizeof(float));
    assert(tmp->kvCache);
    assert(tmp->attentionScores);
    return tmp;
}

void ReleaseRuntimeData(RuntimeData_t *data)
{
    if (data)
    {
        free(data->logits);
        free(data->x);
        free(data->residuals);
        free(data->tmp1);
        free(data->tmp2);
        free(data->perLayerEmbeddings);
        free(data->allLayerModelProjections);
        free(data->downProjected);
        free(data->ffnHiddenGate);
        free(data->ffnHiddenUp);
        free(data->q);
        free(data->k);
        free(data->v);
        free(data->kvCache);
        free(data->kvCacheOffsets);
        free(data->attentionScores);
        free(data->vMixed);
        free(data);
    }
}

static void Dequantize(float *output, size_t outputSize, const GGUF_TensorInfo_t *input, size_t inputOffset)
{
    if (input->type == GGML_TYPE_Q8_0)
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
        for (size_t blockIndex = 0; blockIndex < blockCount; blockIndex++)
        {
            const GGUF_Q8_0_t *block = input->data.q8_0 + inputOffset + blockIndex;
            const float scale = GGUF_Float16ToFloat(block->scale);
            for (size_t quantizedIndex = 0; quantizedIndex < blockSize; quantizedIndex++)
            {
                output[blockIndex * blockSize + quantizedIndex] = scale * (float)block->quantized[quantizedIndex];
            }
        }
    }
    else
    {
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
    DEBUG_VECTOR("attn_norm", runtimeData->tmp1, model->embeddingLength);

    /******************************************************************************
     * Compute Query, Key and Value Vectors for Current Token
     ******************************************************************************/
    // General preparations
    assert(layerWeights->attn_q);
    assert(layerWeights->attn_k);
    assert(layerWeights->attn_v);
    assert(layerWeights->attn_q->dimensionCount == 2);
    assert(layerWeights->attn_k->dimensionCount == 2);
    assert(layerWeights->attn_v->dimensionCount == 2);
    assert(layerWeights->attn_k_norm->dimensionCount == 1);
    const size_t qDimension = layerWeights->attn_q->dimensions[1];
    const size_t kDimension = layerWeights->attn_k->dimensions[1];
    const size_t vDimension = layerWeights->attn_v->dimensions[1];
    const size_t headDimension = layerWeights->attn_k_norm->dimensions[0];
    const size_t headCount = qDimension / headDimension;
    // TODO: Remove hardcoded values --> load these constants from model data.
    const bool globalAttention = (headDimension == 512);
    const size_t sharedKVLayerCount = 20;
    const float ropeBase = globalAttention ? 1000000.0f : 10000.0f;
    assert(layerWeights->attn_q_norm->type == GGML_TYPE_F32);
    assert(layerWeights->attn_q_norm->dimensions[0] == headDimension);

    const size_t firstLayerWithSharedKV = model->blockCount - sharedKVLayerCount;
    const size_t swaSharedKVLayer = firstLayerWithSharedKV - 2;
    const size_t globalSharedKVLayer = firstLayerWithSharedKV - 1;

    if (blockNumber < firstLayerWithSharedKV)
    {
        // Apply projection matrices
        MultiplyMatrixAndVector(runtimeData->k, kDimension, runtimeData->tmp1, model->embeddingLength, layerWeights->attn_k);
        MultiplyMatrixAndVector(runtimeData->v, vDimension, runtimeData->tmp1, model->embeddingLength, layerWeights->attn_v);
        DEBUG_VECTOR("Kcur-0", runtimeData->k, kDimension);
        DEBUG_VECTOR("Vcur-0", runtimeData->v, vDimension);

        RMSNorm(runtimeData->k, runtimeData->k, kDimension);
        RMSNorm(runtimeData->v, runtimeData->v, vDimension);
        assert(layerWeights->attn_k_norm->type == GGML_TYPE_F32);
        assert(layerWeights->attn_k_norm->dimensions[0] == kDimension);
        MultiplyFloats(runtimeData->k, runtimeData->k, layerWeights->attn_k_norm->data.float32, kDimension);
        DEBUG_VECTOR("", runtimeData->k, kDimension);

        // RoPE
        ApplyRoPE(runtimeData->k, kDimension, headDimension, position, ropeBase);
        DEBUG_VECTOR("", runtimeData->q, qDimension);
        DEBUG_VECTOR("", runtimeData->k, kDimension);
    }
    else
    {
        // Layers >= 15 reuse the KV cache values from layer 13/14
        const size_t anchorLayer = globalAttention ? globalSharedKVLayer : swaSharedKVLayer;
        float *kvCacheShared = runtimeData->kvCache + runtimeData->kvCacheOffsets[anchorLayer];

        float *kvCacheSharedForToken = kvCacheShared + position * (kDimension + vDimension);
        memcpy(runtimeData->k, kvCacheSharedForToken, kDimension * sizeof(float));
        memcpy(runtimeData->v, kvCacheSharedForToken + kDimension, vDimension * sizeof(float));
    }

    MultiplyMatrixAndVector(runtimeData->q, qDimension, runtimeData->tmp1, model->embeddingLength, layerWeights->attn_q);
    DEBUG_VECTOR("Qcur-0", runtimeData->q, qDimension);
    for (size_t head = 0; head < headCount; head++)
    {
        const size_t offset = head * headDimension;
        RMSNorm(runtimeData->q + offset, runtimeData->q + offset, headDimension);
        MultiplyFloats(runtimeData->q + offset, runtimeData->q + offset, layerWeights->attn_q_norm->data.float32, headDimension);
        DEBUG_VECTOR("", (runtimeData->q + offset), headDimension);
    }
    DEBUG_VECTOR("Qcur-0", runtimeData->q, qDimension);
    ApplyRoPE(runtimeData->q, qDimension, headDimension, position, ropeBase);
    DEBUG_VECTOR("Qcur-0", runtimeData->q, qDimension);

    /******************************************************************************
     * Update KV Cache
     ******************************************************************************/
    // TODO: I think it might be better to have seperate buffers for the K and V values.
    float *kvCacheForLayer = runtimeData->kvCache + runtimeData->kvCacheOffsets[blockNumber];
    float *kvCacheForToken = kvCacheForLayer + position * (kDimension + vDimension);
    memcpy(kvCacheForToken, runtimeData->k, kDimension * sizeof(float));
    memcpy(kvCacheForToken + kDimension, runtimeData->v, vDimension * sizeof(float));

    /******************************************************************************
     * Blend Cached Values
     ******************************************************************************/
    const size_t startPosition = globalAttention ? 0 : (position > 512 ? position - 512 : 0);

    for (size_t head = 0; head < headCount; head++)
    {
        // Measure similarity between the query and all cached keys
        const float *qHead = runtimeData->q + head * headDimension;
        const size_t kvHeadCount = kDimension / headDimension;
        const size_t kvHead = head / (headCount / kvHeadCount);
        for (size_t tokenPos = startPosition; tokenPos <= position; tokenPos++)
        {
            const float *tokenKey = runtimeData->kvCache + runtimeData->kvCacheOffsets[blockNumber] + tokenPos * (kDimension + vDimension) + (kvHead * headDimension);
            runtimeData->attentionScores[tokenPos] = DotProduct(qHead, tokenKey, headDimension); // / sqrtf(headDimension);
        }

        // Apply softmax to the scores (note: initial implementation had numerical stability issues).
        float maxScore = -INFINITY;
        for (size_t tokenPos = startPosition; tokenPos <= position; tokenPos++)
        {
            if (runtimeData->attentionScores[tokenPos] > maxScore)
            {
                maxScore = runtimeData->attentionScores[tokenPos];
            }
        }

        float sum = 0.0f;
        for (size_t tokenPos = startPosition; tokenPos <= position; tokenPos++)
        {
            runtimeData->attentionScores[tokenPos] = expf(runtimeData->attentionScores[tokenPos] - maxScore);
            sum += runtimeData->attentionScores[tokenPos];
        }
        for (size_t tokenPos = startPosition; tokenPos <= position; tokenPos++)
        {
            runtimeData->attentionScores[tokenPos] /= sum;
        }

        // Merge values according to the similarity between query and keys.
        float *outputHead = runtimeData->vMixed + head * headDimension;
        memset(outputHead, 0, headDimension * sizeof(float));
        for (size_t tokenPos = startPosition; tokenPos <= position; tokenPos++)
        {
            const float *valueInCache = runtimeData->kvCache + runtimeData->kvCacheOffsets[blockNumber] + tokenPos * (kDimension + vDimension) + kDimension + (kvHead * headDimension);
            AddFloatsScaled(outputHead, outputHead, valueInCache, runtimeData->attentionScores[tokenPos], headDimension);
        }
    }
    for (size_t head = 0; head < headCount; head++)
    {

        DEBUG_VECTOR("", (runtimeData->vMixed + headDimension * head), vDimension);
    }

    /******************************************************************************
     * Finalize Attention Output
     ******************************************************************************/
    // Project back down to the hidden dimension
    MultiplyMatrixAndVector(runtimeData->tmp2, model->embeddingLength, runtimeData->vMixed, qDimension, layerWeights->attn_output);
    DEBUG_VECTOR("", runtimeData->tmp2, model->embeddingLength);

    // Normalization
    RMSNorm(runtimeData->tmp2, runtimeData->tmp2, model->embeddingLength);
    assert(layerWeights->post_attention_norm->type == GGML_TYPE_F32);
    assert(layerWeights->post_attention_norm->dimensions[0] == model->embeddingLength);
    MultiplyFloats(runtimeData->tmp2, runtimeData->tmp2, layerWeights->post_attention_norm->data.float32, model->embeddingLength);
    // Add back original scaled token embedding
    AddFloats(runtimeData->tmp1, runtimeData->tmp2, runtimeData->x, model->embeddingLength);
    DEBUG_VECTOR("", runtimeData->tmp1, model->embeddingLength);
}

static void RunFeedForward(Model_t *model, RuntimeData_t *runtimeData, uint32_t blockNumber)
{
    const BlockWeights_t *layerWeights = model->weights.blocks + blockNumber;

    /******************************************************************************
     * Normalization
     ******************************************************************************/
    RMSNorm(runtimeData->tmp2, runtimeData->tmp1, model->embeddingLength);
    assert(layerWeights->ffn_norm->type == GGML_TYPE_F32);
    assert(layerWeights->ffn_norm->dimensions[0] == model->embeddingLength);
    MultiplyFloats(runtimeData->tmp2, runtimeData->tmp2, layerWeights->ffn_norm->data.float32, model->embeddingLength);
    DEBUG_VECTOR("", runtimeData->tmp2, model->embeddingLength);

    /******************************************************************************
     * Feed Forward Network
     ******************************************************************************/
    // Expand to the hidden dimension, apply the activation function and then project back down.
    // The size differs by layer. Buffers ffnHiddenGate and ffnHiddenUp are allocated for the largest case.
    assert(layerWeights->ffn_gate);
    assert(layerWeights->ffn_up);
    assert(layerWeights->ffn_down);
    assert(layerWeights->ffn_gate->dimensionCount == 2);
    const size_t hiddenDimension = layerWeights->ffn_gate->dimensions[1];
    MultiplyMatrixAndVector(runtimeData->ffnHiddenGate, hiddenDimension, runtimeData->tmp2, model->embeddingLength, layerWeights->ffn_gate);
    MultiplyMatrixAndVector(runtimeData->ffnHiddenUp, hiddenDimension, runtimeData->tmp2, model->embeddingLength, layerWeights->ffn_up);
    DEBUG_VECTOR("", runtimeData->ffnHiddenGate, hiddenDimension);
    DEBUG_VECTOR("", runtimeData->ffnHiddenUp, hiddenDimension);

    // Apply GEGLU activation function
    for (size_t i = 0; i < hiddenDimension; i++)
    {
        const float x = runtimeData->ffnHiddenGate[i];
        runtimeData->ffnHiddenGate[i] = (x * 0.5f * (1.0f + erff(x / 1.41421356f))) * runtimeData->ffnHiddenUp[i];
    }
    DEBUG_VECTOR("", runtimeData->ffnHiddenGate, hiddenDimension);

    // Down projection
    MultiplyMatrixAndVector(runtimeData->tmp2, model->embeddingLength, runtimeData->ffnHiddenGate, hiddenDimension, layerWeights->ffn_down);
    DEBUG_VECTOR("", runtimeData->tmp2, model->embeddingLength);

    /******************************************************************************
     * Finalize FFN Ouput
     ******************************************************************************/
    // Normalization
    RMSNorm(runtimeData->tmp2, runtimeData->tmp2, model->embeddingLength);
    assert(layerWeights->post_ffw_norm->type == GGML_TYPE_F32);
    assert(layerWeights->post_ffw_norm->dimensions[0] == model->embeddingLength);
    MultiplyFloats(runtimeData->tmp2, runtimeData->tmp2, layerWeights->post_ffw_norm->data.float32, model->embeddingLength);
    // Add back in the original input (= attention output)
    AddFloats(runtimeData->tmp2, runtimeData->tmp2, runtimeData->tmp1, model->embeddingLength);
    DEBUG_VECTOR("", runtimeData->tmp2, model->embeddingLength);
}

static void RunInjection(Model_t *model, RuntimeData_t *runtimeData, uint32_t blockNumber)
{
    const BlockWeights_t *layerWeights = model->weights.blocks + blockNumber;

    CopyFloats(runtimeData->residuals, runtimeData->tmp2, model->embeddingLength);

    /******************************************************************************
     * Down Projection
     ******************************************************************************/
    assert(layerWeights->inp_gate);
    assert(layerWeights->inp_gate->dimensionCount == 2);
    const size_t perLayerInjectedSize = layerWeights->inp_gate->dimensions[1];
    MultiplyMatrixAndVector(runtimeData->downProjected, perLayerInjectedSize, runtimeData->tmp2, model->embeddingLength, layerWeights->inp_gate);
    DEBUG_VECTOR("", runtimeData->downProjected, perLayerInjectedSize);

    // GELU activation function
    for (size_t i = 0; i < perLayerInjectedSize; i++)
    {
        const float x = runtimeData->downProjected[i];
        runtimeData->downProjected[i] = (x * 0.5f * (1.0f + erff(x / 1.41421356f)));
    }
    DEBUG_VECTOR("", runtimeData->downProjected, perLayerInjectedSize);

    /******************************************************************************
     * Inject Embedding
     ******************************************************************************/
    const size_t offset = blockNumber * perLayerInjectedSize;
    float *currentContext = runtimeData->tmp1; // Reusing the first 256 indices of tmp1
    RMSNorm(currentContext, runtimeData->allLayerModelProjections + offset, perLayerInjectedSize);
    assert(model->weights.per_layer_proj_norm);
    assert(model->weights.per_layer_proj_norm->dimensionCount == 1);
    assert(model->weights.per_layer_proj_norm->dimensions[0] == perLayerInjectedSize);
    assert(model->weights.per_layer_proj_norm->type == GGML_TYPE_F32);
    MultiplyFloats(currentContext, currentContext, model->weights.per_layer_proj_norm->data.float32, perLayerInjectedSize);

    AddFloats(currentContext, currentContext, runtimeData->perLayerEmbeddings + offset, perLayerInjectedSize);
    MultiplyFloats(runtimeData->downProjected, runtimeData->downProjected, currentContext, perLayerInjectedSize);
    DEBUG_VECTOR("", runtimeData->downProjected, perLayerInjectedSize);

    /******************************************************************************
     * Project back up
     ******************************************************************************/
    assert(layerWeights->proj);
    MultiplyMatrixAndVector(runtimeData->tmp1, model->embeddingLength, runtimeData->downProjected, perLayerInjectedSize, layerWeights->proj);
    DEBUG_VECTOR("", runtimeData->tmp1, model->embeddingLength);

    RMSNorm(runtimeData->tmp1, runtimeData->tmp1, model->embeddingLength);
    assert(layerWeights->post_norm->type == GGML_TYPE_F32);
    assert(layerWeights->post_norm->dimensions[0] == model->embeddingLength);
    MultiplyFloats(runtimeData->tmp1, runtimeData->tmp1, layerWeights->post_norm->data.float32, model->embeddingLength);
    DEBUG_VECTOR("", runtimeData->tmp1, model->embeddingLength);

    AddFloats(runtimeData->tmp1, runtimeData->tmp2, runtimeData->tmp1, model->embeddingLength);
    assert(layerWeights->layer_output_scale);
    assert(layerWeights->layer_output_scale->type == GGML_TYPE_F32);
    assert(layerWeights->layer_output_scale->dimensions[0] == 1);
    ScaleFloats(runtimeData->x, runtimeData->tmp1, layerWeights->layer_output_scale->data.float32[0], model->embeddingLength);
    DEBUG_VECTOR("", runtimeData->x, model->embeddingLength);
}

static float *RunClassifier(Model_t *model, RuntimeData_t *runtimeData)
{

    RMSNorm(runtimeData->x, runtimeData->x, model->embeddingLength);
    assert(model->weights.output_norm);
    assert(model->weights.output_norm->type == GGML_TYPE_F32);
    assert(model->weights.output_norm->dimensions[0] == model->embeddingLength);
    MultiplyFloats(runtimeData->x, runtimeData->x, model->weights.output_norm->data.float32, model->embeddingLength);
    DEBUG_VECTOR("", runtimeData->x, model->embeddingLength);

    MultiplyMatrixAndVector(runtimeData->logits, model->tokenCount, runtimeData->x, model->embeddingLength, model->weights.token_embd);

    // TODO: load value from 'gemma4.final_logit_softcapping' = 30.000000
    for (size_t i = 0; i < model->tokenCount; i++)
    {
        runtimeData->logits[i] = 30.0f * tanhf(runtimeData->logits[i] / 30.0f);
    }
    DEBUG_VECTOR("", runtimeData->logits, model->tokenCount);
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

static void AddFloatsScaled(float *out, const float *in, const float *added, float scale, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = in[i] + scale * added[i];
    }
}

static void RMSNorm(float *output, const float *input, size_t count)
{
    double sum = 0.0;
    for (size_t i = 0; i < count; i++)
    {
        sum += (double)input[i] * input[i];
    }
    // TODO: Load epsilon from model file
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
    else if (matrix->type == GGML_TYPE_F16)
    {
        for (size_t row = 0; row < rows; row++)
        {
            float dotProduct = 0.0f;
            for (size_t col = 0; col < cols; col++)
            {
                const float weight = GGUF_Float16ToFloat(matrix->data.float16[cols * row + col]);
                dotProduct += input[col] * weight;
            }
            out[row] = dotProduct;
        }
    }
    else
    {
        printf("Unknown quantized type: %d\n", matrix->type);
        assert(false);
    }
}

static void ApplyRoPE(float *vector, size_t count, size_t headSize, uint32_t position, float frequencyBase)
{
    assert(count % headSize == 0);
    const size_t headCount = count / headSize;
    const size_t halfDimension = headSize / 2;
    for (size_t head = 0; head < headCount; head++)
    {
        float *headData = vector + head * headSize;
        for (size_t i = 0; i < halfDimension; i++)
        {
            // TODO: I think there is a tensor that already contains the frequency values, although I don't
            //       understand how this is supposed to work with the different attention layers (global / local).
            const float theta = (float)position / powf(frequencyBase, (float)(2 * i) / headSize);
            const float cosTheta = cosf(theta);
            const float sinTheta = sinf(theta);
            const float x = headData[i];
            const float y = headData[i + halfDimension];

            headData[i] = x * cosTheta - y * sinTheta;
            headData[i + halfDimension] = x * sinTheta + y * cosTheta;
        }
    }
}
static float DotProduct(const float *a, const float *b, size_t count)
{
    float sum = 0.0f;
    for (size_t i = 0; i < count; i++)
    {
        sum += a[i] * b[i];
    }
    return sum;
}
