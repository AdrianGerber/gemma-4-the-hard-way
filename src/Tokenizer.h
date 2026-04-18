/**
 * @file      Tokenizer.h
 * @author    Adrian Gerber
 * @brief     Extremely slow implementation of a tokenizer :).
 *            Converts between human-readable text and tokens that the LLM understands. For now, the main (known) limitations are the following:
 *            - Bad performance scaling with longer input strings (naive implementation).
 *            - Some edge cases with special characters are not handled properly (e.g. multi-byte characters that don't appear in the LLM token list).
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

#ifndef TOKENIZER_H_
#define TOKENIZER_H_

/******************************************************************************
 * Includes
 ******************************************************************************/
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include "GGUF.h"

/******************************************************************************
 * Constants and Macros
 ******************************************************************************/

/******************************************************************************
 * Type Definitions
 ******************************************************************************/

/**
 * @brief Structure holding all the information needed for tokenization.
 *
 */
typedef struct
{
    GGUF_MetadataValueArray_t tokens;     // Each entry defines the string of a token, with the index being the token ID.
    GGUF_MetadataValueArray_t scores;     // Not used for now (alternative way of tokenizing)
    GGUF_MetadataValueArray_t tokenTypes; // Identification of special token types (e.g. control sequences)
    GGUF_MetadataValueArray_t merges;     // Rules for how to prioritize token merging. Lower indices take priority.
    uint32_t tokenIdBos, tokenIdEos, tokenIdUnknown, tokenIdPadding, tokenIdMask, tokenEndOfTurn;
} Tokenizer_t;

/**
 * @brief Result of the tokenization process.
 *
 */
typedef struct
{
    size_t length;    // Number of tokens in the array
    uint32_t *tokens; // Dynamically allocated array of token IDs.
} TokenizerEncoded_t;

/******************************************************************************
 * Public Function Prototypes
 ******************************************************************************/

/**
 * @brief Setup a tokenizer from a model's metadata.
 *
 * @param metadata Array of metadata entries.
 * @param count Length of array.
 * @return Tokenizer_t Initialized tokenizer.
 */
Tokenizer_t Tokenizer_Init(GGUF_Metadata_t *metadata, size_t count);

/**
 * @brief Encode a string using the available tokens.
 *
 * @param tokenizer Instance.
 * @param input String to encode. Can contain special characters.
 * @return TokenizerEncoded_t Resulting vector of tokens. Has to be free'd using Tokenizer_ReleaseEncoded.
 */
TokenizerEncoded_t Tokenizer_Encode(Tokenizer_t tokenizer, const char *input);

/**
 * @brief Free a token vector.
 *
 * @param output Token vector to free.
 */
void Tokenizer_ReleaseEncoded(TokenizerEncoded_t input);

/**
 * @brief Print a vector of tokens to stdout.
 *
 * @param tokenizer Tokenizer instance.
 * @param input List of tokens to print.
 * @param groupTokens false: print the resulting raw text, true: Add visual indications for token boundaries.
 */
void Tokenizer_DecodeToStdOut(Tokenizer_t tokenizer, TokenizerEncoded_t input, bool groupTokens);

/**
 * @brief Release any memory allocated by the tokenizer.
 *
 * @param tokenizer Instance.
 */
void Tokenizer_Release(Tokenizer_t tokenizer);

#endif /* TOKENIZER_H_ */
