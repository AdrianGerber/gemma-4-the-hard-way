/**
 * @file      LLMMath.c
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

/******************************************************************************
 * Includes
 ******************************************************************************/
#include "LLMMath.h"
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

/******************************************************************************
 * Private Constants and Macros
 ******************************************************************************/

/******************************************************************************
 * Private Type Definitions
 ******************************************************************************/

/******************************************************************************
 * Private / Variables
 ******************************************************************************/

/******************************************************************************
 * Private Function Prototypes
 ******************************************************************************/

/******************************************************************************
 * Public Function Implementations
 ******************************************************************************/

void AddTensors(float *out, const float *a, const float *b, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = a[i] + b[i];
    }
}

void MultiplyTensors(float *out, const float *a, const float *b, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = a[i] * b[i];
    }
}

void ScaleTensor(float *out, const float *in, float factor, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = factor * in[i];
    }
}

void CopyTensor(float *destination, const float *source, size_t count)
{
    memcpy(destination, source, count * sizeof(float));
}

void AddScaledTensor(float *out, const float *in, const float *added, float scale, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        out[i] = in[i] + scale * added[i];
    }
}

void RMSNorm(float *output, const float *input, float epsilon, size_t count)
{
    const float sumOfSquares = DotProduct(input, input, count);
    const float normalizingFactor = 1.0f / sqrtf(sumOfSquares / count + epsilon);
    ScaleTensor(output, input, normalizingFactor, count);
}

void RMSNormWithWeights(float *output, const float *input, float epsilon, const GGUF_TensorInfo_t *weights, size_t count)
{
    RMSNorm(output, input, epsilon, count);

    // Apply element-wise weights
    assert(weights);
    assert(weights->type == GGML_TYPE_F32);
    assert(weights->dimensionCount == 1);
    assert(weights->dimensions[0] == count);
    MultiplyTensors(output, output, weights->data.float32, count);
}

void MultiplyMatrixAndVector(float *restrict out, size_t outCount, const float *restrict input, size_t inCount, const GGUF_TensorInfo_t *matrix)
{
    assert(matrix);
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
                const float *inputBlock = input + block * blockSize;
                const int8_t *quanized = rowBlocks[block].quantized;

                // Unroll the loop using 4 temporary sums to break the dependency chain and
                // allow better optimizations by the compiler.
                float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
                for (size_t i = 0; i < blockSize; i += 4)
                {
                    sum0 += inputBlock[i + 0] * (float)quanized[i + 0];
                    sum1 += inputBlock[i + 1] * (float)quanized[i + 1];
                    sum2 += inputBlock[i + 2] * (float)quanized[i + 2];
                    sum3 += inputBlock[i + 3] * (float)quanized[i + 3];
                }

                // Save 16 multiplications per block by applying the scale once after each block.
                dotProduct += scale * (sum0 + sum1 + sum2 + sum3);
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
        fprintf(stderr, "Unsupported quantized type: %d\n", matrix->type);
        assert(false);
    }
}

float DotProduct(const float *a, const float *b, size_t count)
{
    float sum = 0.0f;
    for (size_t i = 0; i < count; i++)
    {
        sum += a[i] * b[i];
    }
    return sum;
}

void SoftCap(float *output, const float *input, float weight, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        output[i] = weight * tanhf(input[i] / weight);
    }
}

void SoftMax(float *output, const float *input, size_t count)
{
    // Find the maximum value first. This is then subtracted from all values to avoid numberical stability issues.
    float max = -INFINITY;
    for (size_t i = 0; i < count; i++)
    {
        if (input[i] > max)
        {
            max = input[i];
        }
    }

    float sum = 0.0f;
    for (size_t i = 0; i < count; i++)
    {
        output[i] = expf(input[i] - max);
        sum += output[i];
    }
    for (size_t i = 0; i < count; i++)
    {
        output[i] /= sum;
    }
}

void DequantizeTensor(float *output, size_t outputSize, const GGUF_TensorInfo_t *input, size_t inputOffset)
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
        fprintf(stderr, "Unsupported quantized type: %d\n", input->type);
        assert(false);
    }
}

void ApplyRoPE(float *vector, size_t count, size_t headSize, uint32_t position, float frequencyBase)
{
    assert(count % headSize == 0);
    const size_t headCount = count / headSize;
    const size_t halfDimension = headSize / 2;
    for (size_t head = 0; head < headCount; head++)
    {
        float *headData = vector + head * headSize;
        for (size_t i = 0; i < halfDimension; i++)
        {
            const float theta = (float)position / powf(frequencyBase, (float)(2 * i) / headSize);

            // Apply 2D rotation
            const float cosTheta = cosf(theta);
            const float sinTheta = sinf(theta);
            const float x = headData[i];
            const float y = headData[i + halfDimension];
            headData[i] = x * cosTheta - y * sinTheta;
            headData[i + halfDimension] = x * sinTheta + y * cosTheta;
        }
    }
}

void ApplyGeLu(float *output, float *input, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        const float x = input[i];
        output[i] = (x * 0.5f * (1.0f + erff(x / 1.41421356f)));
    }
}
