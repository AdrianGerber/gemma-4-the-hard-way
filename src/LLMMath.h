/**
 * @file      LLMMath.h
 * @author    Adrian Gerber
 * @brief     Mathematical primitives for the language model.
 *
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

#ifndef LLM_MATH_H_
#define LLM_MATH_H_

/******************************************************************************
 * Includes
 ******************************************************************************/
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "GGUF.h"

/******************************************************************************
 * Constants and Macros
 ******************************************************************************/

/******************************************************************************
 * Type Definitions
 ******************************************************************************/

/******************************************************************************
 * Public Function Prototypes
 ******************************************************************************/

/**
 * @brief Element-wise addition of two lists of floats.
 *
 * @param out Result is written to this location. Can be one of the inputs.
 * @param a List a
 * @param b List b
 * @param count Number of elements to add.
 */
void AddTensors(float *out, const float *a, const float *b, size_t count);

/**
 * @brief Element-wise multiplication of two lists of floats.
 *
 * @param out Result is written to this location. Can be one of the inputs.
 * @param a List a
 * @param b List b
 * @param count Number of elements to multiply.
 */
void MultiplyTensors(float *out, const float *a, const float *b, size_t count);

/**
 * @brief Multiply a list of floats by a constant factor.
 *
 * @param out Result is written to this location. Can be one the input.
 * @param in Values to scale
 * @param factor Scaling factor.
 * @param count Number of elements to scale.
 */
void ScaleTensor(float *out, const float *in, float factor, size_t count);

/**
 * @brief Copy a list of floats.
 *
 * @param destination Floats are written to this address.
 * @param source Floats are read from this address.
 * @param count Number of elements to copy.
 */
void CopyTensor(float *destination, const float *source, size_t count);

/**
 * @brief Scale a list of floats by a constant factor and perform element-wise addition to a second list.
 *
 * @param out Result is written to this location. Can be one of the inputs.
 * @param in First list of floats
 * @param added List of floats that is scaled and then added to the input list.
 * @param scale Scaling factor applied to the 'added' list.
 * @param count Number of elements to process.
 */
void AddScaledTensor(float *out, const float *in, const float *added, float scale, size_t count);

/**
 * @brief Perform RMS normalization.
 *
 * @param output Result is written to this location. Can be the input.
 * @param input Values to be normalized.
 * @param epsilon Epsilon to add for numerical stability.
 * @param count Number of elements to normalize.
 */
void RMSNorm(float *output, const float *input, float epsilon, size_t count);

/**
 * @brief Perform RMS normalization with additional weighting factors.
 *
 * @param output Result is written to this location. Can be the input.
 * @param input Values to be normalized.
 * @param epsilon Epsilon to add for numerical stability.
 * @param weights List of weights applied to the resulting vector.
 * @param count Number of elements to normalize.
 */
void RMSNormWithWeights(float *output, const float *input, float epsilon, const GGUF_TensorInfo_t *weights, size_t count);

/**
 * @brief Multiply a vector of floats by a (possibly quantized) 2D tensor.
 *
 * @param out Result is written to this location. Important: Cannot have any memory overlap with the input.
 * @param outCount Number of elements in the output buffer.
 * @param input Input vector.
 * @param inCount Number of elements in the input buffer.
 * @param matrix Matrix to multiply by.
 */
void MultiplyMatrixAndVector(float *restrict out, size_t outCount, const float *restrict input, size_t inCount, const GGUF_TensorInfo_t *matrix);

/**
 * @brief Compute the dot product between two lists of floats.
 *
 * @param a List a
 * @param b List b
 * @param count Number of elements.
 * @return float Resulting dot product.
 */
float DotProduct(const float *a, const float *b, size_t count);

/**
 * @brief Perform tanh-based softcapping.
 *
 * @param output Result is written to this location. Can be the input.
 * @param input Values to be capped
 * @param weight Softcapping weight.
 * @param count Number of elements to softcap.
 */
void SoftCap(float *output, const float *input, float weight, size_t count);

/**
 * @brief Apply the softmax function to a vector of values.
 *
 * @param output Result is written to this location. Can be the input.
 * @param input Input values.
 * @param count Number of elements to softcap.
 */
void SoftMax(float *output, const float *input, size_t count);

/**
 * @brief Dequantize a tensor into a list of floats.
 *
 * @param output Output buffer.
 * @param outputSize Number of elements in the output buffer.
 * @param input Tensor to dequantize.
 * @param inputOffset Number of elements to skip in the input tensor.
 */
void DequantizeTensor(float *output, size_t outputSize, const GGUF_TensorInfo_t *input, size_t inputOffset);

/**
 * @brief Apply the rotary position embedding to a vector of key or query heads.
 *
 * @param vector List of floats. Is modified in-place.
 * @param count Size of the vector.
 * @param headSize Size of a single head. The number of attention heads is count / headSize.
 * @param position Index of the currently processed token.
 * @param frequencyBase RoPE base frequency.
 */
void ApplyRoPE(float *vector, size_t count, size_t headSize, uint32_t position, float frequencyBase);

/**
 * @brief Apply the GeLu activation function to a vector.
 *
 * @param output Result is written to this location. Can be the input.
 * @param input Input values
 * @param count Number of elements.
 */
void ApplyGeLu(float *output, float *input, size_t count);

#endif /* LLM_MATH_H_ */
