/**
 * @file      Debug.h
 * @author    Adrian Gerber
 * @brief     Debugging helper macros and preprocessor switches.
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

#ifndef DEBUG_H_
#define DEBUG_H_

/******************************************************************************
 * Includes
 ******************************************************************************/
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

/******************************************************************************
 * Constants and Macros
 ******************************************************************************/

// Debug switches
#define DEBUG_TENSOR_VALUES 0
#define DEBUG_TOKEN_PROBABILITIES 0
#define DEBUG_PRINT_TENSORINFO 0
#define DEBUG_PRINT_METADATA 0
#define DEBUG_PRINT_TOKENIZER 0

// Utility macros
#if DEBUG_TENSOR_VALUES
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

// Ugly but very useful debug macro :)
#define DEBUG_TENSOR(vector, length)                                                                                                                                                            \
    {                                                                                                                                                                                           \
        float tmpSum123 = 0.0f;                                                                                                                                                                 \
        for (size_t i = 0; i < length; i++)                                                                                                                                                     \
            tmpSum123 += vector[i];                                                                                                                                                             \
        printf("%04u %-20s %-30s sum=%06f            value=[%f, %f, ..., %f, %f]\n", __LINE__, tag, TOSTRING(vector), tmpSum123, vector[0], vector[1], vector[length - 2], vector[length - 1]); \
    }
#else
#define DEBUG_TENSOR(vector, length)
#endif

#endif /* DEBUG_H_ */
