/**
 * @file      Model.h
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
 * @brief Weights applied during each transformer block
 *
 */
typedef struct
{
    // Attention
    const GGUF_TensorInfo_t *attn_k;
    const GGUF_TensorInfo_t *attn_k_norm;
    const GGUF_TensorInfo_t *attn_norm;
    const GGUF_TensorInfo_t *attn_output;
    const GGUF_TensorInfo_t *attn_q;
    const GGUF_TensorInfo_t *attn_q_norm;
    const GGUF_TensorInfo_t *attn_v;
    const GGUF_TensorInfo_t *post_attention_norm;

    // Feed Forward Network
    const GGUF_TensorInfo_t *ffn_down;
    const GGUF_TensorInfo_t *ffn_gate;
    const GGUF_TensorInfo_t *ffn_norm;
    const GGUF_TensorInfo_t *ffn_up;
    const GGUF_TensorInfo_t *inp_gate;
    const GGUF_TensorInfo_t *post_ffw_norm;

    // Token injection
    const GGUF_TensorInfo_t *post_norm;
    const GGUF_TensorInfo_t *proj;
    const GGUF_TensorInfo_t *layer_output_scale;
} BlockWeights_t;

/**
 * @brief Structure to represent the final probability of each token. Mainly for
 *        convenience so we can apply qsort to find the most likely tokens.
 *
 */
typedef struct
{
    uint32_t tokenId;  // ID of the token
    float probability; // Probability that this token ID is selected.
} TokenProbability_t;

/**
 * @brief Buffers required for the computation (allocated once to avoid repeated malloc/free calls in the hot loop).
 *
 */
typedef struct
{
    // General buffers
    float *x;
    float *residuals;
    float *tmp1, *tmp2;

    // Attention
    float *q;
    float *k;
    float *v;
    float *kvCache;
    size_t *kvCacheOffsets;
    float *attentionScores;
    float *vMixed;

    // Feed Forward Network
    float *ffnHiddenGate;
    float *ffnHiddenUp;

    // Per-layer token injection
    float *perLayerEmbeddings;
    float *allLayerModelProjections;
    float *downProjected;

    // Output
    float *logits;
    TokenProbability_t *tokenProbabilities;
} RuntimeData_t;

/**
 * @brief Structure representing the constant information needed to run a language model.
 *
 */
typedef struct
{
    // Lists of metadata parameters and tensors loaded from the GGUF file
    GGUF_Metadata_t *metadata;
    GGUF_TensorInfo_t *tensorInfo;
    size_t metadataCount, tensorInfoCount;

    // General constants extracted from the metadata for easy access.
    size_t blockCount, tokenCount, contextSize, embeddingLength, alignment;

    // Tokenizer instance set up with the model's vocabulary.
    Tokenizer_t tokenizer;

    // Weights for inference
    struct
    {
        // Per-block (35 blocks for gemma-4-e2b)
        BlockWeights_t *blocks;

        // Token embedding
        const GGUF_TensorInfo_t *token_embd;

        // RMS normalization
        float rmsNormEpsilon;

        // Attention (Interleaved sliding window and global layers)
        size_t sharedAttentionLayerCount;
        float ropeFreqBase;
        float ropeFreqBaseSWA;
        size_t attentionSlidingWindowSize;
        size_t attentionSlidingWindowKeyLength;
        size_t attentionKeyLength;

        // Per-layer token injection
        const GGUF_TensorInfo_t *per_layer_token_embd;
        const GGUF_TensorInfo_t *per_layer_model_proj;
        const GGUF_TensorInfo_t *per_layer_proj_norm;

        // Final token selection
        const GGUF_TensorInfo_t *output_norm;
        float finalLogitSoftcapping;
        size_t topK;
        float topP;
        float temperature;
    } weights;
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
