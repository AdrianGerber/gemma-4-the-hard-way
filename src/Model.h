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

typedef struct
{
    const GGUF_TensorInfo_t *attn_k;
    const GGUF_TensorInfo_t *attn_k_norm;
    const GGUF_TensorInfo_t *attn_norm;
    const GGUF_TensorInfo_t *attn_output;
    const GGUF_TensorInfo_t *attn_q;
    const GGUF_TensorInfo_t *attn_q_norm;
    const GGUF_TensorInfo_t *attn_v;
    const GGUF_TensorInfo_t *ffn_down;
    const GGUF_TensorInfo_t *ffn_gate;
    const GGUF_TensorInfo_t *ffn_norm;
    const GGUF_TensorInfo_t *ffn_up;
    const GGUF_TensorInfo_t *inp_gate;
    const GGUF_TensorInfo_t *layer_output_scale;
    const GGUF_TensorInfo_t *post_attention_norm;
    const GGUF_TensorInfo_t *post_ffw_norm;
    const GGUF_TensorInfo_t *post_norm;
    const GGUF_TensorInfo_t *proj;
} BlockWeights_t;

typedef struct
{
    float *logits;
    float *x;
    float *residuals;
    float *tmp1, *tmp2;
    float *perLayerEmbeddings;
    float *allLayerModelProjections;
    float *downProjected;
    float *ffnHiddenGate;
    float *ffnHiddenUp;
    float *q;
    float *k;
    float *v;
    float *kvCache;
    size_t *kvCacheOffsets;
    float *attentionScores;
    float *vMixed;
} RuntimeData_t;

/**
 * @brief Structure representing the information needed to run a language model.
 *
 */
typedef struct
{
    GGUF_Metadata_t *metadata;
    GGUF_TensorInfo_t *tensorInfo;
    size_t metadataCount, tensorInfoCount, blockCount, tokenCount, contextSize, embeddingLength, alignment;
    Tokenizer_t tokenizer;
    struct
    {
        BlockWeights_t *blocks;
        const GGUF_TensorInfo_t *output_norm;
        const GGUF_TensorInfo_t *per_layer_model_proj;
        const GGUF_TensorInfo_t *per_layer_proj_norm;
        const GGUF_TensorInfo_t *per_layer_token_embd;
        const GGUF_TensorInfo_t *rope_freqs;
        const GGUF_TensorInfo_t *token_embd;
        float rmsNormEpsilon;
        float ropeFreqBase;
        float ropeFreqBaseSWA;
        size_t sharedAttentionLayerCount;
        float finalLogitSoftcapping;
        size_t attentionSlidingWindowSize;
        size_t attentionKeyLength;
        size_t attentionSlidingWindowKeyLength;
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