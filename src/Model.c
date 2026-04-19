/**
 * @file      Model.c
 * @author    Adrian Gerber
 * @brief     Bare-minimum implementation for the gemma-4-e2b-it-Q8_0 language model.
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
#include "LLMMath.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "Debug.h"

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
static float *RunForwardPass(Model_t *model, RuntimeData_t *runtimeData, uint32_t token, uint32_t position, bool preFill);
static void RunAttention(Model_t *model, RuntimeData_t *runtimeData, uint32_t blockNumber, uint32_t position);
static void RunFeedForward(Model_t *model, RuntimeData_t *runtimeData, uint32_t blockNumber);
static void RunInjection(Model_t *model, RuntimeData_t *runtimeData, uint32_t blockNumber);
static float *RunClassifier(Model_t *model, RuntimeData_t *runtimeData);
static uint32_t SelectTokenFromLogits(Model_t *model, RuntimeData_t *runtimeData, float *logits);

// General utility functions
static int CompareTokenProbabilities(const void *a, const void *b);
static RuntimeData_t *AllocateRuntimeData(Model_t *model);
static void ReleaseRuntimeData(RuntimeData_t *data);
static const GGUF_TensorInfo_t *GetTensorForBlock(Model_t *model, size_t blockIndex, const char *tensorName);

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

    Model_t *model = malloc(sizeof(Model_t));
    assert(model);

    /******************************************************************************
     * Load Metadata
     ******************************************************************************/
    model->metadataCount = GGUF_GetMetadataCount(data);
    model->metadata = malloc(model->metadataCount * sizeof(GGUF_Metadata_t));
    assert(model->metadata);

    const uint8_t *readPointer = GGUF_SkipHeader(data);
    for (size_t i = 0; i < model->metadataCount; i++)
    {
        model->metadata[i] = GGUF_MetadataFromMemory(&readPointer);

#if DEBUG_PRINT_METADATA
        // Skip long strings and arrays.
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
#endif
    }

    /******************************************************************************
     * Load Tensors
     ******************************************************************************/
    model->tensorInfoCount = GGUF_GetTensorCount(data);
    model->tensorInfo = malloc(model->tensorInfoCount * sizeof(GGUF_TensorInfo_t));
    assert(model->tensorInfo);
    for (size_t i = 0; i < model->tensorInfoCount; i++)
    {
        model->tensorInfo[i] = GGUF_TensorInfoFromMemory(&readPointer);
#if DEBUG_PRINT_TENSORINFO
        GGUF_TensorInfoPrint(model->tensorInfo[i]);
#endif
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

    /******************************************************************************
     * Prepare Model Blocks
     ******************************************************************************/
    const GGUF_Metadata_t *blockCount = GGUF_MetadataFindByKey(model->metadata, model->metadataCount, "gemma4.block_count");
    assert(blockCount);
    assert(blockCount->type == GGUF_METADATA_VALUE_TYPE_UINT32);
    model->blockCount = blockCount->value.uint32;
    model->weights.blocks = malloc(model->blockCount * sizeof(BlockWeights_t));
#if DEBUG_PRINT_TENSORINFO
    printf("Loading %lu blocks...\n", model->blockCount);
#endif
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
        // Note: Not all weights might be present on all layers. Availability needs to be checked before use.
    }

    model->weights.output_norm = GGUF_TensorFindByName(model->tensorInfo, model->tensorInfoCount, "output_norm.weight");
    model->weights.per_layer_model_proj = GGUF_TensorFindByName(model->tensorInfo, model->tensorInfoCount, "per_layer_model_proj.weight");
    model->weights.per_layer_proj_norm = GGUF_TensorFindByName(model->tensorInfo, model->tensorInfoCount, "per_layer_proj_norm.weight");
    model->weights.per_layer_token_embd = GGUF_TensorFindByName(model->tensorInfo, model->tensorInfoCount, "per_layer_token_embd.weight");
    model->weights.token_embd = GGUF_TensorFindByName(model->tensorInfo, model->tensorInfoCount, "token_embd.weight");
    assert(model->weights.output_norm);
    assert(model->weights.per_layer_model_proj);
    assert(model->weights.per_layer_proj_norm);
    assert(model->weights.per_layer_token_embd);
    assert(model->weights.token_embd);

    /******************************************************************************
     * Tokenizer
     ******************************************************************************/
    printf("\n");
    model->tokenizer = Tokenizer_Init(model->metadata, model->metadataCount);
    model->tokenCount = model->tokenizer.tokens.length;

    /******************************************************************************
     * Apply Model Configuration
     ******************************************************************************/
    // Determine context size
    const GGUF_Metadata_t *contextSize = GGUF_MetadataFindByKey(model->metadata, model->metadataCount, "gemma4.context_length");
    assert(contextSize);
    assert(contextSize->type == GGUF_METADATA_VALUE_TYPE_UINT32);
    model->contextSize = contextSize->value.uint32;
    if (model->contextSize > 2048)
    {
        printf("Limiting context size to 2048 for now.\n");
        model->contextSize = 2048;
    }

    // General model parameters
    const GGUF_Metadata_t *embeddingLength = GGUF_MetadataFindByKey(model->metadata, model->metadataCount, "gemma4.embedding_length");
    assert(embeddingLength);
    assert(embeddingLength->type == GGUF_METADATA_VALUE_TYPE_UINT32);
    model->embeddingLength = embeddingLength->value.uint32;

    const GGUF_Metadata_t *rmsEpsilon = GGUF_MetadataFindByKey(model->metadata, model->metadataCount, "gemma4.attention.layer_norm_rms_epsilon");
    model->weights.rmsNormEpsilon = 0.000001f;
    if (rmsEpsilon)
    {
        assert(rmsEpsilon->type == GGUF_METADATA_VALUE_TYPE_FLOAT32);
        model->weights.rmsNormEpsilon = rmsEpsilon->value.float32;
    }

    // Token selection
    const GGUF_Metadata_t *topP = GGUF_MetadataFindByKey(model->metadata, model->metadataCount, "general.sampling.top_p");
    const GGUF_Metadata_t *topK = GGUF_MetadataFindByKey(model->metadata, model->metadataCount, "general.sampling.top_k");
    const GGUF_Metadata_t *temperature = GGUF_MetadataFindByKey(model->metadata, model->metadataCount, "general.sampling.temp");
    assert(topP);
    assert(topK);
    assert(temperature);
    assert(topK->type == GGUF_METADATA_VALUE_TYPE_INT32);
    model->weights.topK = topK->value.uint32;
    assert(topP->type == GGUF_METADATA_VALUE_TYPE_FLOAT32);
    model->weights.topP = topP->value.float32;
    assert(temperature->type == GGUF_METADATA_VALUE_TYPE_FLOAT32);
    model->weights.temperature = temperature->value.float32;

    // RoPE
    const GGUF_Metadata_t *ropeFreqBase = GGUF_MetadataFindByKey(model->metadata, model->metadataCount, "gemma4.rope.freq_base");
    const GGUF_Metadata_t *ropeFreqBaseSWA = GGUF_MetadataFindByKey(model->metadata, model->metadataCount, "gemma4.rope.freq_base_swa");
    assert(ropeFreqBase);
    assert(ropeFreqBaseSWA);
    assert(ropeFreqBase->type == GGUF_METADATA_VALUE_TYPE_FLOAT32 && ropeFreqBaseSWA->type == GGUF_METADATA_VALUE_TYPE_FLOAT32);
    model->weights.ropeFreqBaseSWA = ropeFreqBaseSWA->value.float32;
    model->weights.ropeFreqBase = ropeFreqBase->value.float32;

    // Interleaved attention layers
    const GGUF_Metadata_t *sharedAttentionLayerCount = GGUF_MetadataFindByKey(model->metadata, model->metadataCount, "gemma4.attention.shared_kv_layers");
    const GGUF_Metadata_t *finalLogitSoftcapping = GGUF_MetadataFindByKey(model->metadata, model->metadataCount, "gemma4.final_logit_softcapping");
    const GGUF_Metadata_t *attentionSlidingWindowSize = GGUF_MetadataFindByKey(model->metadata, model->metadataCount, "gemma4.attention.sliding_window");
    const GGUF_Metadata_t *attentionKeyLength = GGUF_MetadataFindByKey(model->metadata, model->metadataCount, "gemma4.attention.key_length");
    const GGUF_Metadata_t *attentionSlidingWindowKeyLength = GGUF_MetadataFindByKey(model->metadata, model->metadataCount, "gemma4.attention.key_length_swa");
    assert(sharedAttentionLayerCount && finalLogitSoftcapping && attentionSlidingWindowSize && attentionKeyLength && attentionSlidingWindowKeyLength);
    assert(sharedAttentionLayerCount->type == GGUF_METADATA_VALUE_TYPE_UINT32);
    assert(attentionSlidingWindowSize->type == GGUF_METADATA_VALUE_TYPE_UINT32);
    assert(attentionKeyLength->type == GGUF_METADATA_VALUE_TYPE_UINT32);
    assert(attentionSlidingWindowKeyLength->type == GGUF_METADATA_VALUE_TYPE_UINT32);
    assert(finalLogitSoftcapping->type == GGUF_METADATA_VALUE_TYPE_FLOAT32);
    model->weights.sharedAttentionLayerCount = (size_t)sharedAttentionLayerCount->value.uint32;
    model->weights.finalLogitSoftcapping = finalLogitSoftcapping->value.float32;
    model->weights.attentionSlidingWindowSize = (size_t)attentionSlidingWindowSize->value.uint32;
    model->weights.attentionKeyLength = (size_t)attentionKeyLength->value.uint32;
    model->weights.attentionSlidingWindowKeyLength = (size_t)attentionSlidingWindowKeyLength->value.uint32;
    return model;
}

void Model_GenerateCompletionsToStdOut(Model_t *model, const char *prompt)
{
#if DEBUG_PRINT_TOKENIZER
    printf("Encoding Text '%s'\n", prompt);
#endif

    // Allocate memory for the computations. Doing this here avoids repeated malloc/free
    // calls while the model iterates over the tokens.
    RuntimeData_t *runtimeData = AllocateRuntimeData(model);
    assert(runtimeData);

    // Tokenize the prompt.
    TokenizerEncoded_t tokenIds = Tokenizer_Encode(model->tokenizer, prompt);
    printf("Tokenized: ");
    Tokenizer_DecodeToStdOut(model->tokenizer, tokenIds, true);
    printf("\n");

    // Prefill phase to build the KV cache.
    printf("Prefilling KV Cache...\n");
    uint32_t position = 0;
    for (size_t inputTokenIndex = 0; inputTokenIndex < tokenIds.length - 1; inputTokenIndex++)
    {
        // No output is generated yet.
        (void)RunForwardPass(model, runtimeData, tokenIds.tokens[inputTokenIndex], position++, true);
    }

    // Generate the first new token from the last input token.
    printf("Generating Predictions:\n\n");
    float *logits = RunForwardPass(model, runtimeData, tokenIds.tokens[position], position, false);
    assert(logits);
    uint32_t token = SelectTokenFromLogits(model, runtimeData, logits);
#if !DEBUG_TOKEN_PROBABILITIES
    Tokenizer_DecodeToStdOut(model->tokenizer, (TokenizerEncoded_t){.length = 1, .tokens = &token}, false);
    fflush(stdout);
#endif
    position++;

    // Start predicting more future tokens by feeding the model's output back into itself.
    while ((token != model->tokenizer.tokenIdEos) && (token != model->tokenizer.tokenEndOfTurn))
    {
        logits = RunForwardPass(model, runtimeData, token, position, false);
        assert(logits);
        token = SelectTokenFromLogits(model, runtimeData, logits);
#if !DEBUG_TOKEN_PROBABILITIES
        Tokenizer_DecodeToStdOut(model->tokenizer, (TokenizerEncoded_t){.length = 1, .tokens = &token}, false);
        fflush(stdout);
#endif
        position++;

        if (position >= model->contextSize)
        {
            printf("\n Context size exceeded! Stopping.\n");
            break;
        }
    }
    printf("\n");

    // Cleanup
    ReleaseRuntimeData(runtimeData);
    Tokenizer_ReleaseEncoded(tokenIds);
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
 * Language Model Processing
 ******************************************************************************/

static float *RunForwardPass(Model_t *model, RuntimeData_t *runtimeData, uint32_t token, uint32_t position, bool preFill)
{
    // Embed the new token. The token is sent to the model as the initial state, but also
    // fed back in partially on each layer. For this there exist two different embeddings
    // that we need to load:

    // Main token embedding
    assert(model->weights.token_embd->dimensionCount == 2);
    assert(model->weights.token_embd->dimensions[0] == model->embeddingLength);
    assert(model->weights.token_embd->dimensions[1] > token);
    DequantizeTensor(runtimeData->x, model->embeddingLength, model->weights.token_embd, token * model->embeddingLength);
    ScaleTensor(runtimeData->x, runtimeData->x, sqrtf((float)model->embeddingLength), model->embeddingLength);
    DEBUG_TENSOR(runtimeData->x, model->embeddingLength);

    // Per-layer token embeddings. These are successively injected during the attention
    // block of each layer. This helps the model remember the original token while
    // information passes through the layers. We prepare all the data at this point,
    // but each layer will only inject 256 bytes at a time.
    assert(model->weights.per_layer_token_embd->dimensionCount == 2);
    assert(model->weights.per_layer_token_embd->dimensions[1] > token);
    const size_t injectedEmbeddingSizeTotal = model->weights.per_layer_token_embd->dimensions[0];
    assert(injectedEmbeddingSizeTotal % model->blockCount == 0);
    const size_t injectedEmbeddingSizePerLayer = injectedEmbeddingSizeTotal / model->blockCount;
    DequantizeTensor(runtimeData->perLayerEmbeddings, model->weights.per_layer_token_embd->dimensions[0], model->weights.per_layer_token_embd, token * model->weights.per_layer_token_embd->dimensions[0]);
    ScaleTensor(runtimeData->perLayerEmbeddings, runtimeData->perLayerEmbeddings, sqrtf((float)injectedEmbeddingSizePerLayer), injectedEmbeddingSizeTotal);
    DEBUG_TENSOR(runtimeData->perLayerEmbeddings, injectedEmbeddingSizeTotal);
    MultiplyMatrixAndVector(runtimeData->allLayerModelProjections, injectedEmbeddingSizeTotal, runtimeData->x, model->embeddingLength, model->weights.per_layer_model_proj);
    ScaleTensor(runtimeData->allLayerModelProjections, runtimeData->allLayerModelProjections, 1.0f / sqrtf(model->embeddingLength), injectedEmbeddingSizeTotal);
    DEBUG_TENSOR(runtimeData->allLayerModelProjections, injectedEmbeddingSizeTotal);

    // Run the blocks defined by the LLM
    for (size_t blockNumber = 0; blockNumber < model->blockCount; blockNumber++)
    {
#if DEBUG_TENSOR_VALUES
        printf("\n--- Layer %lu ---\n", blockNumber);
#endif
        RunAttention(model, runtimeData, blockNumber, position);
        RunFeedForward(model, runtimeData, blockNumber);
        RunInjection(model, runtimeData, blockNumber);
    }

    // The logits are only required when the model actually needs to predict a token. Not performing
    // this computation during the prefill phase skips a few huge matrix multiplications.
    float *logits = NULL;
    if (!preFill)
    {
        logits = RunClassifier(model, runtimeData);
    }

    return logits;
}

static void RunAttention(Model_t *model, RuntimeData_t *runtimeData, uint32_t blockNumber, uint32_t position)
{
    const BlockWeights_t *layerWeights = model->weights.blocks + blockNumber;

    RMSNormWithWeights(runtimeData->xTmp, runtimeData->x, model->weights.rmsNormEpsilon, layerWeights->attn_norm, model->embeddingLength);
    DEBUG_TENSOR(runtimeData->xTmp, model->embeddingLength);

    /******************************************************************************
     * Prepare Constants
     ******************************************************************************/
    // Determine QKV attention dimensions for this layer.
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

    // Verify that the assumed head dimension holds true.
    assert(layerWeights->attn_q_norm->type == GGML_TYPE_F32);
    assert(layerWeights->attn_q_norm->dimensions[0] == headDimension);

    // There are two interleaved types of attention layers in the Gemma 4 model. Every 5th block
    // runs global attention with head size 512, while everything else uses SWA with head size 256.
    // - Global attention: Runs against all tokens in the context.
    // - Sliding window attention: Operates just on the last 512 tokens in the context window.
    const bool globalAttention = (headDimension == model->weights.attentionKeyLength);
    // RoPE base frequency must be selected depending on the current block's attention type
    const float ropeBase = globalAttention ? model->weights.ropeFreqBase : model->weights.ropeFreqBaseSWA;

    /******************************************************************************
     * Compute Key and Value Vectors
     ******************************************************************************/
    // Starting from block 15, the KV cache is shared. Global attention layers must access
    // block 14's cache, while SWA layers use block 13. This preserves the dimensions.
    const size_t firstLayerWithSharedKV = model->blockCount - model->weights.sharedAttentionLayerCount;
    const size_t swaSharedKVBlock = firstLayerWithSharedKV - 2;
    const size_t globalSharedKVBlock = firstLayerWithSharedKV - 1;
    const bool accessSharedKVCache = (blockNumber >= firstLayerWithSharedKV);
    size_t cacheBlockIndex = blockNumber;
    if (accessSharedKVCache)
    {
        cacheBlockIndex = globalAttention ? globalSharedKVBlock : swaSharedKVBlock;
    }

    float *kvCache = runtimeData->kvCache + runtimeData->kvCacheOffsets[cacheBlockIndex];

    // Layers >= 15 access the KV values directly from the layer 13/14 cache. This means that
    // we can just skip these calculations on later layers and don't need to copy anything.
    if (!accessSharedKVCache)
    {
        // For layers without cache reuse (block < 15), we have to calculate both K and V.
        // This is done by projecting the model state onto the 'key' and 'value' vectors.
        MultiplyMatrixAndVector(runtimeData->k, kDimension, runtimeData->xTmp, model->embeddingLength, layerWeights->attn_k);
        MultiplyMatrixAndVector(runtimeData->v, vDimension, runtimeData->xTmp, model->embeddingLength, layerWeights->attn_v);
        DEBUG_TENSOR(runtimeData->k, kDimension);
        DEBUG_TENSOR(runtimeData->v, vDimension);

        // Normalization and Scaling
        RMSNorm(runtimeData->k, runtimeData->k, model->weights.rmsNormEpsilon, kDimension);
        RMSNorm(runtimeData->v, runtimeData->v, model->weights.rmsNormEpsilon, vDimension);
        assert(layerWeights->attn_k_norm->type == GGML_TYPE_F32);
        assert(layerWeights->attn_k_norm->dimensions[0] == kDimension);
        MultiplyTensors(runtimeData->k, runtimeData->k, layerWeights->attn_k_norm->data.float32, kDimension);
        DEBUG_TENSOR(runtimeData->k, kDimension);

        // Rotary Position Embedding
        ApplyRoPE(runtimeData->k, kDimension, headDimension, position, ropeBase);
        DEBUG_TENSOR(runtimeData->q, qDimension);
        DEBUG_TENSOR(runtimeData->k, kDimension);

        // Update Cache
        float *kvCacheForToken = kvCache + position * (kDimension + vDimension);
        memcpy(kvCacheForToken, runtimeData->k, kDimension * sizeof(float));
        memcpy(kvCacheForToken + kDimension, runtimeData->v, vDimension * sizeof(float));
    }

    /******************************************************************************
     * Compute Query Vector
     ******************************************************************************/
    // The query vector is also projected from the model state. It indicates what keys in the
    // KV cache are the most relevant to the current state.
    MultiplyMatrixAndVector(runtimeData->q, qDimension, runtimeData->xTmp, model->embeddingLength, layerWeights->attn_q);
    DEBUG_TENSOR(runtimeData->q, qDimension);

    // We are dealing with multi-head attention so the query consists of multiple heads. In essence, the model
    // can search the KV cache for multiple different keys at the same time. Each head needs to be normalized independently.
    for (size_t head = 0; head < headCount; head++)
    {
        const size_t offset = head * headDimension;
        RMSNormWithWeights(runtimeData->q + offset, runtimeData->q + offset, model->weights.rmsNormEpsilon, layerWeights->attn_q_norm, headDimension);
        DEBUG_TENSOR((runtimeData->q + offset), headDimension);
    }

    // Rotary Position Embedding
    // Since we rotated the key, the query needs to be rotated in the same way. The rotation amount is proportional to the current token position.
    // This means that later tokens will be rotated more. In my understanding, this results in key and query from tokens closer together in position
    // to be more similar than ones further apart, prioritizing more recent information while not completely ignoring older values in the KV cache.
    ApplyRoPE(runtimeData->q, qDimension, headDimension, position, ropeBase);
    DEBUG_TENSOR(runtimeData->q, qDimension);

    /******************************************************************************
     * Blend Cached Values
     ******************************************************************************/
    // Global attention starts from 0 every time, while Sliding Window Attention
    // only considers the cache at most 512 tokens into the past.
    size_t startPosition = 0;
    if (!globalAttention && position > model->weights.attentionSlidingWindowSize)
    {
        startPosition = position - model->weights.attentionSlidingWindowSize;
    }

    // Run the search for each individual head in the query vector.
    for (size_t head = 0; head < headCount; head++)
    {
        // Measure similarity between the query and all cached keys
        const float *qHead = runtimeData->q + head * headDimension;
        const size_t kvHeadCount = kDimension / headDimension;
        const size_t kvHead = head / (headCount / kvHeadCount);
        for (size_t tokenPos = startPosition; tokenPos <= position; tokenPos++)
        {
            const float *tokenKey = kvCache + tokenPos * (kDimension + vDimension) + (kvHead * headDimension);
            runtimeData->attentionScores[tokenPos] = DotProduct(qHead, tokenKey, headDimension); // / sqrtf(headDimension); <-- This is a mistake that cost me like 4 hours to find :)
        }

        // Softmax
        SoftMax(runtimeData->attentionScores + startPosition, runtimeData->attentionScores + startPosition, position - startPosition + 1);

        // Add up all the cached values while scaling them. Values are weighted
        // based on the similarity between the model's query and the value's key.
        float *outputHead = runtimeData->vMixed + head * headDimension;
        memset(outputHead, 0, headDimension * sizeof(float));
        for (size_t tokenPos = startPosition; tokenPos <= position; tokenPos++)
        {
            const size_t offsetForValue = tokenPos * (kDimension + vDimension) + kDimension + (kvHead * headDimension);
            const float *valueInCache = kvCache + offsetForValue;
            AddScaledTensor(outputHead, outputHead, valueInCache, runtimeData->attentionScores[tokenPos], headDimension);
        }
        DEBUG_TENSOR((runtimeData->vMixed + headDimension * head), vDimension);
    }

    /******************************************************************************
     * Finalize Attention Output
     ******************************************************************************/
    // Project back to the model state
    MultiplyMatrixAndVector(runtimeData->xTmp, model->embeddingLength, runtimeData->vMixed, qDimension, layerWeights->attn_output);
    DEBUG_TENSOR(runtimeData->xTmp, model->embeddingLength);

    // Normalization
    RMSNormWithWeights(runtimeData->xTmp, runtimeData->xTmp, model->weights.rmsNormEpsilon, layerWeights->post_attention_norm, model->embeddingLength);
    DEBUG_TENSOR(runtimeData->xTmp, model->embeddingLength);

    // Add back original scaled token embedding
    AddTensors(runtimeData->x, runtimeData->xTmp, runtimeData->x, model->embeddingLength);
    DEBUG_TENSOR(runtimeData->x, model->embeddingLength);
}

static void RunFeedForward(Model_t *model, RuntimeData_t *runtimeData, uint32_t blockNumber)
{
    const BlockWeights_t *layerWeights = model->weights.blocks + blockNumber;

    /******************************************************************************
     * Normalization
     ******************************************************************************/
    RMSNormWithWeights(runtimeData->xTmp, runtimeData->x, model->weights.rmsNormEpsilon, layerWeights->ffn_norm, model->embeddingLength);
    DEBUG_TENSOR(runtimeData->xTmp, model->embeddingLength);

    /******************************************************************************
     * Feed Forward Network
     ******************************************************************************/
    // Expand to the hidden dimension, apply the activation function and gate, then project back down.

    // The size differs by layer. Buffers ffnHiddenGate and ffnHiddenUp are allocated for the largest case.
    assert(layerWeights->ffn_gate);
    assert(layerWeights->ffn_up);
    assert(layerWeights->ffn_down);
    assert(layerWeights->ffn_gate->dimensionCount == 2);
    const size_t hiddenDimension = layerWeights->ffn_gate->dimensions[1];

    // Up projection
    MultiplyMatrixAndVector(runtimeData->ffnHiddenGate, hiddenDimension, runtimeData->xTmp, model->embeddingLength, layerWeights->ffn_gate);
    MultiplyMatrixAndVector(runtimeData->ffnHiddenUp, hiddenDimension, runtimeData->xTmp, model->embeddingLength, layerWeights->ffn_up);
    DEBUG_TENSOR(runtimeData->ffnHiddenGate, hiddenDimension);
    DEBUG_TENSOR(runtimeData->ffnHiddenUp, hiddenDimension);

    // The gate vector goes through the activation function. This clamps negative values
    // to roughly zero while leaving positive values roughly unchanged.
    ApplyGeLu(runtimeData->ffnHiddenGate, runtimeData->ffnHiddenGate, hiddenDimension);

    // The 'gate' vector is then multiplied by the 'up' vector, eliminating any elements of the 'up' vector where
    // the corresponding gate value was negative (turned to zero by the activation function). This controls what
    // information is kept before transforming back down.
    MultiplyTensors(runtimeData->ffnHiddenGate, runtimeData->ffnHiddenGate, runtimeData->ffnHiddenUp, hiddenDimension);
    DEBUG_TENSOR(runtimeData->ffnHiddenGate, hiddenDimension);

    // Down projection
    MultiplyMatrixAndVector(runtimeData->xTmp, model->embeddingLength, runtimeData->ffnHiddenGate, hiddenDimension, layerWeights->ffn_down);
    DEBUG_TENSOR(runtimeData->xTmp, model->embeddingLength);

    /******************************************************************************
     * Finalize FFN Ouput
     ******************************************************************************/
    RMSNormWithWeights(runtimeData->xTmp, runtimeData->xTmp, model->weights.rmsNormEpsilon, layerWeights->post_ffw_norm, model->embeddingLength);

    // Add back in the original input (= attention output)
    AddTensors(runtimeData->x, runtimeData->xTmp, runtimeData->x, model->embeddingLength);
    DEBUG_TENSOR(runtimeData->x, model->embeddingLength);
}

static void RunInjection(Model_t *model, RuntimeData_t *runtimeData, uint32_t blockNumber)
{
    const BlockWeights_t *layerWeights = model->weights.blocks + blockNumber;

    /******************************************************************************
     * Down Projection
     ******************************************************************************/
    assert(layerWeights->inp_gate);
    assert(layerWeights->inp_gate->dimensionCount == 2);
    const size_t perLayerInjectedSize = layerWeights->inp_gate->dimensions[1];
    MultiplyMatrixAndVector(runtimeData->downProjected, perLayerInjectedSize, runtimeData->x, model->embeddingLength, layerWeights->inp_gate);
    DEBUG_TENSOR(runtimeData->downProjected, perLayerInjectedSize);

    // Activation function
    ApplyGeLu(runtimeData->downProjected, runtimeData->downProjected, perLayerInjectedSize);
    DEBUG_TENSOR(runtimeData->downProjected, perLayerInjectedSize);

    /******************************************************************************
     * Inject Embedding
     ******************************************************************************/
    const size_t offset = blockNumber * perLayerInjectedSize;
    float *currentContext = runtimeData->xTmp; // Reusing the first 256 indices of xTmp
    RMSNormWithWeights(currentContext, runtimeData->allLayerModelProjections + offset, model->weights.rmsNormEpsilon, model->weights.per_layer_proj_norm, perLayerInjectedSize);

    AddTensors(currentContext, currentContext, runtimeData->perLayerEmbeddings + offset, perLayerInjectedSize);

    MultiplyTensors(runtimeData->downProjected, runtimeData->downProjected, currentContext, perLayerInjectedSize);
    DEBUG_TENSOR(runtimeData->downProjected, perLayerInjectedSize);

    /******************************************************************************
     * Project back up
     ******************************************************************************/
    assert(layerWeights->proj);
    MultiplyMatrixAndVector(runtimeData->xTmp, model->embeddingLength, runtimeData->downProjected, perLayerInjectedSize, layerWeights->proj);
    DEBUG_TENSOR(runtimeData->xTmp, model->embeddingLength);

    RMSNormWithWeights(runtimeData->xTmp, runtimeData->xTmp, model->weights.rmsNormEpsilon, layerWeights->post_norm, model->embeddingLength);
    DEBUG_TENSOR(runtimeData->xTmp, model->embeddingLength);

    AddTensors(runtimeData->xTmp, runtimeData->x, runtimeData->xTmp, model->embeddingLength);
    assert(layerWeights->layer_output_scale);
    assert(layerWeights->layer_output_scale->type == GGML_TYPE_F32);
    assert(layerWeights->layer_output_scale->dimensions[0] == 1);
    ScaleTensor(runtimeData->x, runtimeData->xTmp, layerWeights->layer_output_scale->data.float32[0], model->embeddingLength);
    DEBUG_TENSOR(runtimeData->x, model->embeddingLength);
}

static float *RunClassifier(Model_t *model, RuntimeData_t *runtimeData)
{
    // Final normalization of the model state
    RMSNormWithWeights(runtimeData->x, runtimeData->x, model->weights.rmsNormEpsilon, model->weights.output_norm, model->embeddingLength);
    DEBUG_TENSOR(runtimeData->x, model->embeddingLength);

    // Generate token predictions. This is done by multiplying the model state with the LM Head
    // matrix (262'144 x 1'536). Gemma 4 uses "weight tying", so in this case, LM head is identical
    // to the matrix of token embeddings.
    MultiplyMatrixAndVector(runtimeData->logits, model->tokenCount, runtimeData->x, model->embeddingLength, model->weights.token_embd);
    DEBUG_TENSOR(runtimeData->logits, model->tokenCount);

    // Softcap the final result. There seem to be two primary reasons for this:
    // - Keep values in a reasonable range to prevent floating point overflows (especially with e.g. float16).
    // - Prevent gradients from vanishing during training.
    SoftCap(runtimeData->logits, runtimeData->logits, model->weights.finalLogitSoftcapping, model->tokenCount);
    DEBUG_TENSOR(runtimeData->logits, model->tokenCount);
    return runtimeData->logits;
}

static uint32_t SelectTokenFromLogits(Model_t *model, RuntimeData_t *runtimeData, float *logits)
{
    /******************************************************************************
     * Scaling and Sorting
     ******************************************************************************/
    ScaleTensor(logits, logits, 1.0f / model->weights.temperature, model->tokenCount);
    SoftMax(logits, logits, model->tokenCount);

    // Move to dedicated array for qsort
    for (size_t i = 0; i < model->tokenCount; i++)
    {
        runtimeData->tokenProbabilities[i].tokenId = i;
        runtimeData->tokenProbabilities[i].probability = logits[i];
    }

    // Sort by probability
    qsort(runtimeData->tokenProbabilities, model->tokenCount, sizeof(TokenProbability_t), CompareTokenProbabilities);

    /******************************************************************************
     * Cutoff (top_k / top_p)
     ******************************************************************************/
    float probabilitySum = 0.0f;
    for (size_t i = 0; i < model->tokenCount; i++)
    {
        probabilitySum += runtimeData->tokenProbabilities[i].probability;
    }

    // Determine cutoff (whichever limit of top_k or top_p hits first)
    size_t tokensBeforeCutoff = 0;
    float probabilitySumTopTokens = 0.0f;
    for (size_t i = 0; i < model->weights.topK; i++)
    {
        // Normalization so the entire vocabulary sums to 1.0f
        runtimeData->tokenProbabilities[i].probability /= probabilitySum;

        // Find the cutoff defined by top_p to prevent extremely unlikely tokens from ever getting selected.
        probabilitySumTopTokens += runtimeData->tokenProbabilities[i].probability;
        tokensBeforeCutoff++;
        if (probabilitySumTopTokens >= model->weights.topP)
        {
            break;
        }
    }

    /******************************************************************************
     * Random Sampling
     ******************************************************************************/
    // Normalize the top tokens so their probabilities sum to 1.0f
    for (size_t i = 0; i < tokensBeforeCutoff; i++)
    {
        runtimeData->tokenProbabilities[i].probability /= probabilitySumTopTokens;
    }

    // Select a random element according to the relative probabilities
    float random = (float)rand() / RAND_MAX;
    size_t selectedIndex = tokensBeforeCutoff - 1;
    float accumulator = 0.0f;
    for (size_t i = 0; i < tokensBeforeCutoff; i++)
    {
        accumulator += runtimeData->tokenProbabilities[i].probability;
        if (accumulator >= random)
        {
            selectedIndex = i;
            break;
        }
    }

#if DEBUG_TOKEN_PROBABILITIES
    // Print the probabilities of all tokens that could've been chosen by the random sampling.
    printf("\nConsidered tokens:\n");
    for (size_t i = 0; i < tokensBeforeCutoff; i++)
    {
        printf("- %.1f%% ", 100.0f * runtimeData->tokenProbabilities[i].probability);
        Tokenizer_DecodeToStdOut(model->tokenizer, (TokenizerEncoded_t){.length = 1, .tokens = &runtimeData->tokenProbabilities[i].tokenId}, true);
        if (i == selectedIndex)
        {
            printf(" <-- selected!");
        }
        printf("\n");
    }
    printf("\n");
#endif

    return runtimeData->tokenProbabilities[selectedIndex].tokenId;
}

/******************************************************************************
 * General Utility Functions
 ******************************************************************************/

static int CompareTokenProbabilities(const void *a, const void *b)
{
    // Comparison function according to qsort requirements.
    const TokenProbability_t *tokenA = (const TokenProbability_t *)a;
    const TokenProbability_t *tokenB = (const TokenProbability_t *)b;

    if (tokenA->probability < tokenB->probability)
    {
        return 1;
    }
    if (tokenA->probability > tokenB->probability)
    {
        return -1;
    }
    return 0;
}

static RuntimeData_t *AllocateRuntimeData(Model_t *model)
{
    RuntimeData_t *tmp = malloc(sizeof(RuntimeData_t));
    assert(tmp);

    // General buffers
    tmp->x = malloc(model->embeddingLength * sizeof(float));
    tmp->xTmp = malloc(model->embeddingLength * sizeof(float));
    assert(tmp->x && tmp->xTmp);

    // Perpare attention constants
    // Determine the total size needed to hold the variable-width attention vectors for each layer.
    // Attention layers >= 15 just read the cache from layer 13/14 and don't need to be considered
    // when allocating the cache.
    size_t maxQ = 0;
    size_t maxK = 0;
    size_t maxV = 0;
    const size_t blocksWithOwnKVCache = model->blockCount - model->weights.sharedAttentionLayerCount;
    tmp->kvCacheOffsets = malloc(blocksWithOwnKVCache * sizeof(size_t));
    assert(tmp->kvCacheOffsets);
    size_t kvCacheSize = 0;
    for (size_t block = 0; block < blocksWithOwnKVCache; block++)
    {
        // Store the offset of each layer's cache for later access.
        tmp->kvCacheOffsets[block] = kvCacheSize;

        // Find maximum dimensions
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

        // Keep track of size
        kvCacheSize += model->contextSize * (k->dimensions[1] + v->dimensions[1]);
    }

    // Prepare attention buffers
    tmp->q = malloc(maxQ * sizeof(float));
    tmp->k = malloc(maxK * sizeof(float));
    tmp->v = malloc(maxV * sizeof(float));
    tmp->vMixed = malloc(maxQ * sizeof(float));
    assert(tmp->q && tmp->k && tmp->v && tmp->vMixed);

    // KV cache
    tmp->kvCache = malloc(kvCacheSize * sizeof(float));
    tmp->attentionScores = malloc(model->contextSize * sizeof(float));
    assert(tmp->kvCache && tmp->attentionScores);

    // Feed Forward Network
    // Gemma 4 uses variable dimensions on the feed forward layers. Buffers need to be allocated for the largest layer.
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

    // Per-layer token injection
    assert(model->weights.per_layer_token_embd);
    const size_t totalInjectedSize = model->weights.per_layer_token_embd->dimensions[0];
    const size_t perLayerSize = totalInjectedSize / model->blockCount;
    tmp->perLayerEmbeddings = malloc(totalInjectedSize * sizeof(float));
    tmp->allLayerModelProjections = malloc(totalInjectedSize * sizeof(float));
    tmp->downProjected = malloc(perLayerSize * sizeof(float));
    assert(tmp->perLayerEmbeddings);
    assert(tmp->allLayerModelProjections);
    assert(tmp->downProjected);

    // Output
    tmp->logits = malloc(model->tokenCount * sizeof(float));
    tmp->tokenProbabilities = malloc(model->tokenCount * sizeof(TokenProbability_t));
    assert(tmp->logits);
    assert(tmp->tokenProbabilities);
    return tmp;
}

void ReleaseRuntimeData(RuntimeData_t *data)
{
    if (data)
    {
        free(data->logits);
        free(data->x);
        free(data->xTmp);
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
        free(data->tokenProbabilities);
        free(data);
    }
}

static const GGUF_TensorInfo_t *GetTensorForBlock(Model_t *model, size_t blockIndex, const char *tensorName)
{
    // e.g. blk.34.proj.weight
    char fullName[256];
    assert((int)sizeof(fullName) > snprintf(fullName, sizeof(fullName), "blk.%lu.%s", blockIndex, tensorName));
    return GGUF_TensorFindByName(model->tensorInfo, model->tensorInfoCount, fullName);
}
