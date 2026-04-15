/**
 * @file      Tokenizer.c
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

/******************************************************************************
 * Includes
 ******************************************************************************/
#include "GGUF.h"
#include "Tokenizer.h"
#include <string.h>
#include <stdio.h>
#include <assert.h>
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
static uint32_t StringToToken(Tokenizer_t tokenizer, const char *input, size_t length);
static uint32_t ScorePotentialMerge(Tokenizer_t tokenizer, const char *left, size_t leftLength, const char *right, size_t rightLength);
static char *PreprocessString(const char *input);

/******************************************************************************
 * Public Function Implementations
 ******************************************************************************/

Tokenizer_t Tokenizer_Init(GGUF_Metadata_t *metadata, size_t count)
{
    Tokenizer_t tokenizer;

    // Get all the required values from the model metadata and verify the basic assumptions made by the tokenizer.

    const GGUF_Metadata_t *model = GGUF_MetadataFindByKey(metadata, count, "tokenizer.ggml.model");
    assert(model);
    assert(model->type == GGUF_METADATA_VALUE_TYPE_STRING);
    printf("Setting up tokenizer for %s:\n", model->value.str);

    const GGUF_Metadata_t *tokens = GGUF_MetadataFindByKey(metadata, count, "tokenizer.ggml.tokens");
    assert(tokens);
    assert(tokens->type == GGUF_METADATA_VALUE_TYPE_ARRAY);
    assert(tokens->value.array.type == GGUF_METADATA_VALUE_TYPE_STRING);
    tokenizer.tokens = tokens->value.array;
    printf("- Found %lu tokens.\n", tokenizer.tokens.length);

    const GGUF_Metadata_t *scores = GGUF_MetadataFindByKey(metadata, count, "tokenizer.ggml.scores");
    assert(scores);
    assert(scores->type == GGUF_METADATA_VALUE_TYPE_ARRAY);
    assert(scores->value.array.type == GGUF_METADATA_VALUE_TYPE_FLOAT32);
    tokenizer.scores = scores->value.array;
    printf("- Found %lu scores.\n", tokenizer.scores.length);
    assert(tokenizer.scores.length == tokenizer.tokens.length);

    const GGUF_Metadata_t *tokenTypes = GGUF_MetadataFindByKey(metadata, count, "tokenizer.ggml.token_type");
    assert(tokenTypes);
    assert(tokenTypes->type == GGUF_METADATA_VALUE_TYPE_ARRAY);
    assert(tokenTypes->value.array.type == GGUF_METADATA_VALUE_TYPE_INT32);
    tokenizer.tokenTypes = tokenTypes->value.array;
    printf("- Found %lu token types.\n", tokenizer.tokenTypes.length);
    assert(tokenizer.tokenTypes.length == tokenizer.tokens.length);

    const GGUF_Metadata_t *merges = GGUF_MetadataFindByKey(metadata, count, "tokenizer.ggml.merges");
    assert(merges);
    assert(merges->type == GGUF_METADATA_VALUE_TYPE_ARRAY);
    assert(merges->value.array.type == GGUF_METADATA_VALUE_TYPE_STRING);
    tokenizer.merges = merges->value.array;
    printf("- Found %lu merge rules.\n", tokenizer.merges.length);

    const GGUF_Metadata_t *tokenIdBos = GGUF_MetadataFindByKey(metadata, count, "tokenizer.ggml.bos_token_id");
    const GGUF_Metadata_t *tokenIdEos = GGUF_MetadataFindByKey(metadata, count, "tokenizer.ggml.eos_token_id");
    const GGUF_Metadata_t *tokenIdUnknown = GGUF_MetadataFindByKey(metadata, count, "tokenizer.ggml.unknown_token_id");
    const GGUF_Metadata_t *tokenIdPadding = GGUF_MetadataFindByKey(metadata, count, "tokenizer.ggml.padding_token_id");
    const GGUF_Metadata_t *tokenIdMask = GGUF_MetadataFindByKey(metadata, count, "tokenizer.ggml.mask_token_id");
    assert(tokenIdBos);
    assert(tokenIdEos);
    assert(tokenIdUnknown);
    assert(tokenIdPadding);
    assert(tokenIdMask);
    assert(tokenIdBos->type == GGUF_METADATA_VALUE_TYPE_UINT32);
    assert(tokenIdEos->type == GGUF_METADATA_VALUE_TYPE_UINT32);
    assert(tokenIdUnknown->type == GGUF_METADATA_VALUE_TYPE_UINT32);
    assert(tokenIdPadding->type == GGUF_METADATA_VALUE_TYPE_UINT32);
    assert(tokenIdMask->type == GGUF_METADATA_VALUE_TYPE_UINT32);
    tokenizer.tokenIdBos = tokenIdBos->value.uint32;
    tokenizer.tokenIdEos = tokenIdEos->value.uint32;
    tokenizer.tokenIdUnknown = tokenIdUnknown->value.uint32;
    tokenizer.tokenIdPadding = tokenIdPadding->value.uint32;
    tokenizer.tokenIdMask = tokenIdMask->value.uint32;
    printf("- Loaded special tokens (bos=%u, eos=%u, unknown=%u, padding=%u, mask=%u)\n",
           tokenizer.tokenIdBos,
           tokenizer.tokenIdEos,
           tokenizer.tokenIdUnknown,
           tokenizer.tokenIdPadding,
           tokenizer.tokenIdMask);

    const GGUF_Metadata_t *addSpacePrefix = GGUF_MetadataFindByKey(metadata, count, "tokenizer.ggml.add_space_prefix");
    const GGUF_Metadata_t *addBosToken = GGUF_MetadataFindByKey(metadata, count, "tokenizer.ggml.add_bos_token");
    assert(addSpacePrefix);
    assert(addBosToken);
    assert(addSpacePrefix->type == GGUF_METADATA_VALUE_TYPE_BOOL);
    assert(addBosToken->type == GGUF_METADATA_VALUE_TYPE_BOOL);
    assert(addSpacePrefix->value.bool_ == false);
    assert(addBosToken->value.bool_ == false);

    printf("\n");
    return tokenizer;
}

TokenizerEncoded_t Tokenizer_Encode(Tokenizer_t tokenizer, const char *input)
{
    // Performance measurement
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);

    TokenizerEncoded_t output = {
        .length = 0,
        .tokens = NULL,
    };

    // The string has to be preprocessed to format it in a way that matches the LLM's training.
    // E.g. spaces need to be replaced with a special character.
    char *processedInput = PreprocessString(input);

    // Representation of the tokenizer state during merging of adjacent tokens:
    typedef struct
    {
        uint32_t tokenIndex; // Corresponding token in the LLM tokens list
        size_t length;       // Nubmer of input bytes mapping to this token.
        bool scored;         // Did we already compute the score for this token?
        uint32_t mergeScore; // Cached score for merging to the right.
    } TmpToken_t;
    const size_t initialCount = strlen(processedInput);
    size_t finalTokenCount = initialCount;

    // I'm sure there are smart ways to reduce the amount of memory required to represent the initial tokens,
    // but I doubt that it would matter here especially compared to the memory footprint of the full LLM computations.
    // For now, each TmpToken_t in the array maps to the input string by character index.
    TmpToken_t *tmp = calloc(initialCount, sizeof(TmpToken_t));
    assert(tmp);

    // Initial split
    for (size_t i = 0; i < initialCount;)
    {
        // Treat multi byte UTF-8 characters as a single initial token
        if (((uint8_t)(processedInput[i]) & 0x80) == 0)
            tmp[i].length = 1;
        else if (((uint8_t)(processedInput[i]) & 0xE0) == 0xC0)
            tmp[i].length = 2;
        else if (((uint8_t)(processedInput[i]) & 0xF0) == 0xE0)
            tmp[i].length = 3;
        else if (((uint8_t)(processedInput[i]) & 0xF8) == 0xF0)
            tmp[i].length = 4;

        assert(initialCount - i >= tmp[i].length);

        finalTokenCount -= (tmp[i].length - 1);
        // Note: I think if a multi-byte character ends up as an unknown token, we are supposed to break it down into individual bytes.
        //       This is not implemented yet.
        tmp[i].tokenIndex = StringToToken(tokenizer, processedInput + i, tmp[i].length);
        i += tmp[i].length;
    }

    // Find and tokenize special control tokens in a first pass. Assumes that there
    // is no ambiguity possible in the control tokens.
    for (size_t i = 0; i < tokenizer.tokenTypes.length; i++)
    {
        // Skip non-control tokens
        if (tokenizer.tokenTypes.array[i].uint32 != 3)
            continue;

        // Tokenize all instances of this control string.
        const char *specialTokenStr = tokenizer.tokens.array[i].str;
        const size_t specialTokenLength = strlen(specialTokenStr);
        const char *found = strstr(processedInput, specialTokenStr);
        while (found)
        {
            tmp[found - processedInput].length = specialTokenLength;
            tmp[found - processedInput].tokenIndex = i;
            for (size_t clearIndex = 0; clearIndex < specialTokenLength - 1; clearIndex++)
            {
                tmp[(size_t)(found - processedInput) + 1 + clearIndex].length = 0;
            }

            finalTokenCount -= specialTokenLength - 1;

            found = strstr(found + specialTokenLength, specialTokenStr);
        }
    }

    // rank-based BPE tokenizer
    while (finalTokenCount > 1)
    {
        // Note: This loop is inefficient on multiple levels. First of all, both scoring the merges
        //       and converting strings to tokens result in linear searches on huge arrays. There
        //       should probably be some smart datastructure (hash-map?) or at least a binary search
        //       on the sorted arrays.

        // Find the first active token
        size_t leftTokenForMerge = 0;
        while (tmp[leftTokenForMerge].length == 0)
        {
            leftTokenForMerge++;
        }

        // Score all possible merges to find the one that performs the best.
        size_t bestLeftToken = 0;
        uint32_t bestScore = tokenizer.merges.length;
        for (size_t mergeTry = 0; mergeTry < finalTokenCount - 1; mergeTry++)
        {
            // Find the subsequent token
            size_t rightTokenForMerge = leftTokenForMerge + 1;
            while (tmp[rightTokenForMerge].length == 0)
            {
                rightTokenForMerge++;
            }

            // Only recalculate the score for tokens that were affected by a merge
            uint32_t score = tmp[leftTokenForMerge].mergeScore;
            if (!tmp[leftTokenForMerge].scored)
            {
                score = ScorePotentialMerge(tokenizer, processedInput + leftTokenForMerge, tmp[leftTokenForMerge].length, processedInput + rightTokenForMerge, tmp[rightTokenForMerge].length);
                tmp[leftTokenForMerge].scored = true;
                tmp[leftTokenForMerge].mergeScore = score;
            }

            if (score < bestScore) // '<' instead of '<=' to prioritize left-most merges
            {
                bestScore = score;
                bestLeftToken = leftTokenForMerge;
            }

            // next merge candidate
            leftTokenForMerge++;
            while (tmp[leftTokenForMerge].length == 0)
            {
                leftTokenForMerge++;
            }
        }

        // Perform best merge
        if (bestScore != tokenizer.merges.length)
        {
            leftTokenForMerge = bestLeftToken;
            size_t rightTokenForMerge = leftTokenForMerge + 1;
            while (tmp[rightTokenForMerge].length == 0)
            {
                rightTokenForMerge++;
            }
            tmp[leftTokenForMerge].length += tmp[rightTokenForMerge].length;
            tmp[rightTokenForMerge].length = 0;
            tmp[leftTokenForMerge].tokenIndex = StringToToken(tokenizer, processedInput + leftTokenForMerge, tmp[leftTokenForMerge].length);
            // The new token hasn't been scored yet.
            tmp[leftTokenForMerge].scored = false;

            // Because a new token was created, the cached merge score for the token to its left needs to be invalidated.
            if (leftTokenForMerge)
            {
                size_t scoreInvalidatedToken = leftTokenForMerge - 1;
                while (tmp[scoreInvalidatedToken].length == 0 && scoreInvalidatedToken)
                {
                    scoreInvalidatedToken--;
                }
                // Note: In case no active token is present to the left, this has no effect because it will just
                //       write to index 0 of the array (is inactive in such a case).
                tmp[scoreInvalidatedToken].scored = false;
            }

            // We have now managed to represent the input text with one less token.
            finalTokenCount--;
        }
        else
        {
            // Nothing merged. Tokenizer done.
            break;
        }
    }

    // Copy the result into a dedicated output buffer. This allows us to remove any
    // temporary tokens that were deleted / merged during tokenization.
    output.length = finalTokenCount;
    output.tokens = malloc(finalTokenCount * sizeof(uint32_t));
    assert(output.tokens);
    size_t outputIndex = 0;
    for (size_t i = 0; i < initialCount; i++)
    {
        if (tmp[i].length)
        {
            output.tokens[outputIndex] = tmp[i].tokenIndex;
            outputIndex++;
        }
    }

    free(tmp);
    free(processedInput);

    // Statistics to see just how slow the code runs :).
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec + end.tv_nsec / 1000000000.0) - (start.tv_sec + start.tv_nsec / 1000000000.0);
    printf("Tokenizer took %lfs for %lu iterations.\n", elapsed, initialCount - finalTokenCount);
    return output;
}

void Tokenizer_ReleaseEncoded(TokenizerEncoded_t input)
{
    if (input.tokens)
    {
        free(input.tokens);
    }
}

void Tokenizer_DecodeToStdOut(Tokenizer_t tokenizer, TokenizerEncoded_t input, bool groupTokens)
{
    for (size_t i = 0; i < input.length; i++)
    {
        if (groupTokens)
        {
            if (i != 0)
            {
                printf(", ");
            }
            printf("'");
        }

        // Get the raw string representing the token.
        const uint32_t token = input.tokens[i];
        assert(token < tokenizer.tokens.length);
        const char *str = tokenizer.tokens.array[token].str;

        // Print it character by character so we can apply some post-processing to it.
        while (*str)
        {
            // Restore human-readable spaces.
            if ((uint8_t)str[0] == 0xE2 && (uint8_t)str[1] == 0x96 && (uint8_t)str[2] == 0x81)
            {
                printf(" ");
                str += 3;
            }
            else
            {
                // Escape newlines for debugging
                if (groupTokens && (*str == '\n'))
                {
                    printf("\\n");
                }
                // In most cases, we can just print the unmodified byte
                else
                {
                    printf("%c", *str);
                }
                str++;
            }
        }
        if (groupTokens)
        {
            printf("'");
        }
    }
}

void Tokenizer_Release(Tokenizer_t tokenizer)
{
    (void)tokenizer;
}

/******************************************************************************
 * Private Function Implementations
 ******************************************************************************/

static uint32_t StringToToken(Tokenizer_t tokenizer, const char *input, size_t length)
{
    (void)input;
    (void)length;

    // Naive implementation that searches for a given string in the token array. Ideally we would use a more efficient search algorithm.
    char inputNullTerminated[128];
    assert(length < (sizeof(inputNullTerminated) / sizeof(inputNullTerminated[0])) - 1);
    memcpy(inputNullTerminated, input, length);
    inputNullTerminated[length] = '\0';
    for (size_t tokenIndex = 0; tokenIndex < tokenizer.tokens.length; tokenIndex++)
    {
        if (strcmp(inputNullTerminated, tokenizer.tokens.array[tokenIndex].str) == 0)
        {
            return tokenIndex;
        }
    }

    return tokenizer.tokenIdUnknown;
}

static uint32_t ScorePotentialMerge(Tokenizer_t tokenizer, const char *left, size_t leftLength, const char *right, size_t rightLength)
{
    char merge[256];
    assert(leftLength + rightLength + 1 < (sizeof(merge) / sizeof(merge[0])) - 1);

    // Potential merges are represented as two tokens joined by a space.
    // --> We can join the tokens and then search the merges array.
    memcpy(merge, left, leftLength);
    merge[leftLength] = ' ';
    memcpy(merge + leftLength + 1, right, rightLength);
    merge[leftLength + 1 + rightLength] = '\0';

    for (size_t i = 0; i < tokenizer.merges.length; i++)
    {
        if (strcmp(tokenizer.merges.array[i].str, merge) == 0)
            return i;
    }
    return tokenizer.merges.length;
}

static char *PreprocessString(const char *input)
{
    // The resulting output string will be longer than the input because spaces are expanded.
    size_t numberOfSpaces = 0;
    size_t inputLength = 0;
    while (input[inputLength])
    {
        if (input[inputLength] == ' ')
            numberOfSpaces++;
        inputLength++;
    }

    const size_t outputLength = inputLength + 2 * numberOfSpaces + 1;
    char *output = malloc(outputLength);
    assert(output);

    size_t outputIndex = 0;
    for (size_t inputIndex = 0; inputIndex < inputLength; inputIndex++)
    {
        // SentencePiece tokenizer used by Google replaces spaces with U+2581
        if (input[inputIndex] == ' ')
        {
            output[outputIndex++] = (char)0xE2;
            output[outputIndex++] = (char)0x96;
            output[outputIndex++] = (char)0x81;
        }
        else
        {
            output[outputIndex++] = input[inputIndex];
        }
    }

    output[outputLength - 1] = '\0';
    return output;
}
