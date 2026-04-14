/**
 * @file      GGUF.h
 * @author    Adrian Gerber
 * @brief     Bare-minimum implementation for working with GGUF files.
 *            This code comes with the following limitations:
 *            - Only types required by gemma-4-e2b-it-Q8_0 are implemented for now.
 *            - Unsupported types or failing memory allocations are caught by asserts. No recovery attempts are made.
 *            - It is assumed that the file comes from a trusted source, is consistent and not e.g. malformed.
 *            - Do not use this to load potentially malicious files.
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

#ifndef GGUF_H_
#define GGUF_H_

/******************************************************************************
 * Includes
 ******************************************************************************/
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/******************************************************************************
 * Constants and Macros
 ******************************************************************************/

/******************************************************************************
 * Type Definitions
 ******************************************************************************/

/**
 * @brief Store GGUF strings as null-terminated C strings.
 *
 */
typedef char *GGUF_String_t;

/**
 * @brief Types of GGUF metadata fields. Not everything is implemented here.
 *        Enum copied from https://github.com/ggml-org/ggml/blob/master/docs/gguf.md.
 *
 */
typedef enum
{
    // The value is a 8-bit unsigned integer.
    //   GGUF_METADATA_VALUE_TYPE_UINT8 = 0,
    //   // The value is a 8-bit signed integer.
    //   GGUF_METADATA_VALUE_TYPE_INT8 = 1,
    //   // The value is a 16-bit unsigned little-endian integer.
    //   GGUF_METADATA_VALUE_TYPE_UINT16 = 2,
    //   // The value is a 16-bit signed little-endian integer.
    //   GGUF_METADATA_VALUE_TYPE_INT16 = 3,
    //   // The value is a 32-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT32 = 4,
    //   // The value is a 32-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT32 = 5,
    //   // The value is a 32-bit IEEE754 floating point number.
    GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6,
    //   // The value is a boolean.
    //   // 1-byte value where 0 is false and 1 is true.
    //   // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    GGUF_METADATA_VALUE_TYPE_BOOL = 7,
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    GGUF_METADATA_VALUE_TYPE_STRING = 8,
    // The value is an array of other values, with the length and type prepended.
    ///
    // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    GGUF_METADATA_VALUE_TYPE_ARRAY = 9,
    //   // The value is a 64-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT64 = 10,
    //   // The value is a 64-bit signed little-endian integer.
    //   GGUF_METADATA_VALUE_TYPE_INT64 = 11,
    //   // The value is a 64-bit IEEE754 floating point number.
    //   GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12,
} GGUF_MetadataType_t;

typedef union GGUF_MetadataValue_u GGUF_MetadataValue_t;

/**
 * @brief Array types in GGUF metadata.
 *
 */
typedef struct
{
    // Any value type is valid, including arrays.
    GGUF_MetadataType_t type;
    // Number of elements, not bytes
    uint64_t length;
    // The array of values.
    GGUF_MetadataValue_t *array;
} GGUF_MetadataValueArray_t;

/**
 * @brief Union holding all possible valuês of a metadata field.
 *
 */
union GGUF_MetadataValue_u
{
    //  uint8_t uint8;
    //  int8_t int8;
    //  uint16_t uint16;
    //  int16_t int16;
    uint32_t uint32;
    int32_t int32;
    float float32;
    uint64_t uint64;
    // int64_t int64;
    //  double float64;
    bool bool_;
    GGUF_String_t str;
    GGUF_MetadataValueArray_t array;
};

/**
 * @brief Representation of a single GGUF metadata key-value-pair.
 *
 */
typedef struct
{
    GGUF_String_t key;
    GGUF_MetadataType_t type;
    GGUF_MetadataValue_t value;
} GGUF_Metadata_t;

/**
 * @brief Available data types for tensors.
 *        Enum copied from https://github.com/ggml-org/ggml/blob/master/docs/gguf.md.
 *
 */
typedef enum
{
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
    // GGML_TYPE_Q4_0 = 2,
    // GGML_TYPE_Q4_1 = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 = 5, support has been removed
    // GGML_TYPE_Q5_0 = 6,
    // GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    // GGML_TYPE_Q8_1 = 9,
    // GGML_TYPE_Q2_K = 10,
    // GGML_TYPE_Q3_K = 11,
    // GGML_TYPE_Q4_K = 12,
    // GGML_TYPE_Q5_K = 13,
    // GGML_TYPE_Q6_K = 14,
    // GGML_TYPE_Q8_K = 15,
    // GGML_TYPE_IQ2_XXS = 16,
    // GGML_TYPE_IQ2_XS = 17,
    // GGML_TYPE_IQ3_XXS = 18,
    // GGML_TYPE_IQ1_S = 19,
    // GGML_TYPE_IQ4_NL = 20,
    // GGML_TYPE_IQ3_S = 21,
    // GGML_TYPE_IQ2_S = 22,
    // GGML_TYPE_IQ4_XS = 23,
    // GGML_TYPE_I8 = 24,
    // GGML_TYPE_I16 = 25,
    // GGML_TYPE_I32 = 26,
    // GGML_TYPE_I64 = 27,
    // GGML_TYPE_F64 = 28,
    // GGML_TYPE_IQ1_M = 29,
    // GGML_TYPE_BF16 = 30,
    // GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
    // GGML_TYPE_Q4_0_4_8 = 32,
    // GGML_TYPE_Q4_0_8_8 = 33,
    // GGML_TYPE_TQ1_0 = 34,
    // GGML_TYPE_TQ2_0 = 35,
    // GGML_TYPE_IQ4_NL_4_4 = 36,
    // GGML_TYPE_IQ4_NL_4_8 = 37,
    // GGML_TYPE_IQ4_NL_8_8 = 38,
    // GGML_TYPE_MXFP4 = 39, // MXFP4 (1 block)
    // GGML_TYPE_COUNT = 40,
} GGUF_Type_t;

typedef uint16_t float16_t; // This looks so wrong :)

typedef struct
{
    float16_t scale;
    int8_t quantized[32];
} GGUF_Q8_0_t;

/**
 * @brief Union holding the raw weights.
 *
 */
typedef union
{
    const float *float32;
    const float16_t *float16;
    const GGUF_Q8_0_t *q8_0;
} GGUF_TensorData_t;

/**
 * @brief Defines a tensor as stored in a GGUF file.
 *
 */
typedef struct
{
    GGUF_String_t name;
    uint32_t dimensionCount;
    uint64_t *dimensions;
    GGUF_Type_t type;
    GGUF_TensorData_t data;
} GGUF_TensorInfo_t;

/******************************************************************************
 * Public Function Prototypes
 ******************************************************************************/

/**
 * @brief Convert a GGUF string into the null-terminated C format.
 *
 * @param data Pointer to a pointer to the raw GGUF data representing the string.
 *             Is advanced to past the string data.
 * @return GGUF_String_t Returns a malloc'd C string.
 */
GGUF_String_t GGUF_StringFromMemory(const uint8_t **data);

/**
 * @brief Free an allocated string.
 *
 * @param str String instance.
 */
void GGUF_StringRelease(const GGUF_String_t str);

/**
 * @brief Decode the type of a GGUF metadata field.
 *
 * @param data Pointer to a pointer to the raw GGUF data representing the type.
 *             Is advanced to past the string data.
 * @return GGUF_MetadataType_t Enum representing the type.
 */
GGUF_MetadataType_t GGUF_MetadataTypeFromMemory(const uint8_t **data);

/**
 * @brief Decode the value of a metadata field.
 *
 * @param type Type of the value.
 * @param data data Pointer to a pointer to the raw GGUF data representing the value.
 *             Is advanced to past the string data.
 * @return GGUF_MetadataValue_t Union holding the value. Content must be free'd using GGUF_MetadataValueRelease.
 */
GGUF_MetadataValue_t GGUF_MetadataValueFromMemory(GGUF_MetadataType_t type, const uint8_t **data);

/**
 * @brief Free the value of a metadata field.
 *
 * @param type Type of the value.
 * @param value Metadata value union.
 */
void GGUF_MetadataValueRelease(GGUF_MetadataType_t type, GGUF_MetadataValue_t value);

/**
 * @brief Print the metadata value to stdout for debugging.
 *
 * @param type Type of the value.
 * @param value Metadata value union.
 */
void GGUF_MetadataValuePrint(GGUF_MetadataType_t type, GGUF_MetadataValue_t value);

/**
 * @brief Read an entire key-value pair from GGUF metadata.
 *
 * @param data data Pointer to a pointer to the raw GGUF data representing the metadata.
 *             Is advanced to past the string data.
 * @return GGUFMetadata_t Key-value-pair. Must be free'd using GGUFMetadata_Release.
 */
GGUF_Metadata_t GGUF_MetadataFromMemory(const uint8_t **data);

/**
 * @brief Free a metadata key-value-pair.
 *
 * @param metadata Instance.
 */
void GGUF_MetadataRelease(GGUF_Metadata_t metadata);

/**
 * @brief Print a metadata key-value pair to stdout for debugging.
 *
 * @param metadata Instance.
 */
void GGUF_MetadataPrint(GGUF_Metadata_t metadata);

/**
 * @brief Read a tensor info entry from the GGUF file format.
 *
 * @param data data Pointer to a pointer to the raw GGUF data representing the tensor info section.
 *             Is advanced to past the string data.
 * @param startOfFile Pointer to the beginning of the file.
 * @return GGUF_TensorInfo_t Resulting tensor information. Must be free'd using GGUF_TensorInfoRelease.
 */
GGUF_TensorInfo_t GGUF_TensorInfoFromMemory(const uint8_t **data, const uint8_t *startOfFile);

/**
 * @brief Free a tensor information structure.
 *
 * @param info Instance.
 */
void GGUF_TensorInfoRelease(GGUF_TensorInfo_t info);

/**
 * @brief Print a tensor infor struct to stdout for debugging.
 *
 * @param info Instance.
 */
void GGUF_TensorInfoPrint(GGUF_TensorInfo_t info);

/**
 * @brief Find a metada item by name.
 *
 * @param metadata List of all metadata entries.
 * @param count Number of entries in the list.
 * @param key Key to search for.
 * @return const GGUF_Metadata_t* NULL if not found.
 */
const GGUF_Metadata_t *GGUF_MetadataFindByKey(const GGUF_Metadata_t *metadata, size_t count, const char *key);

/**
 * @brief Find a tensor by name.
 *
 * @param tensors List of all tensors.
 * @param count Number of entries in the list.
 * @param key Key to search for.
 * @return const GGUF_TensorInfo_t* NULL if not found.
 */
const GGUF_TensorInfo_t *GGUF_TensorFindByName(const GGUF_TensorInfo_t *tensorInfo, size_t count, const char *key);

/**
 * @brief Check that the GGUF header is present and supported.
 *
 * @param data Raw bytes.
 * @param length Number of bytes.
 * @return true Header correct.
 * @return false Error.
 */
bool GGUF_VerifyHeader(const uint8_t *data, size_t length);

/**
 * @brief Get the number of tensors in a GGUF file.
 *
 * @param data Raw GGUF file bytes.
 * @return uint64_t Number of tensors.
 */
uint64_t GGUF_GetTensorCount(const uint8_t *data);

/**
 * @brief Get the number of metadata key-value pairs in a GGUF file.
 *
 * @param data Raw GGUF file bytes.
 * @return uint64_t Number of metadata key-value pairs.
 */
uint64_t GGUF_GetMetadataCount(const uint8_t *data);

/**
 * @brief Get a pointer to the first byte after the GGUF header.
 *
 * @param data Raw bytes.
 * @return const uint8_t* First byte after the header (start of metadata entries).
 */
const uint8_t *GGUF_SkipHeader(const uint8_t *data);

#endif /* GGUF_H_ */
