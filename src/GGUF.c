/**
 * @file      GGUF.c
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

/******************************************************************************
 * Includes
 ******************************************************************************/
#include "GGUF.h"
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

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

/******************************************************************************
 * Public Function Implementations
 ******************************************************************************/

GGUF_String_t GGUF_StringFromMemory(const uint8_t **data)
{
    const uint64_t length = *((const uint64_t *)(*data));
    *data += 8;

    GGUF_String_t str = malloc(length + 1);
    assert(str);

    memcpy(str, *data, length);
    *data += length;
    str[length] = '\0';
    return str;
}

void GGUF_StringRelease(const GGUF_String_t str)
{
    if (str)
        free(str);
}

GGUF_MetadataType_t GGUF_MetadataTypeFromMemory(const uint8_t **data)
{
    GGUF_MetadataType_t tmp = *((const uint32_t *)(*data));
    *data += sizeof(uint32_t);
    return tmp;
}

GGUF_MetadataValue_t GGUF_MetadataValueFromMemory(GGUF_MetadataType_t type, const uint8_t **data)
{
    GGUF_MetadataValue_t value;
    switch (type)
    {
    case GGUF_METADATA_VALUE_TYPE_STRING:
        value.str = GGUF_StringFromMemory(data);
        break;
    case GGUF_METADATA_VALUE_TYPE_BOOL:
        value.bool_ = *((const uint8_t *)(*data));
        *data += sizeof(uint8_t);
        break;
    case GGUF_METADATA_VALUE_TYPE_INT32:
        value.int32 = *((const int32_t *)(*data));
        *data += sizeof(int32_t);
        break;
    case GGUF_METADATA_VALUE_TYPE_UINT32:
        value.uint32 = *((const uint32_t *)(*data));
        *data += sizeof(uint32_t);
        break;
    case GGUF_METADATA_VALUE_TYPE_UINT64:
        value.uint64 = *((const uint64_t *)(*data));
        *data += sizeof(uint64_t);
        break;
    case GGUF_METADATA_VALUE_TYPE_FLOAT32:
        value.float32 = *((const float *)(*data));
        *data += sizeof(float);
        break;
    case GGUF_METADATA_VALUE_TYPE_ARRAY:
        value.array.type = GGUF_MetadataTypeFromMemory(data);
        value.array.length = GGUF_MetadataValueFromMemory(GGUF_METADATA_VALUE_TYPE_UINT64, data).uint64;
        value.array.array = malloc(value.array.length * sizeof(GGUF_MetadataValue_t));
        assert(value.array.array);

        for (size_t i = 0; i < value.array.length; i++)
        {
            value.array.array[i] = GGUF_MetadataValueFromMemory(value.array.type, data);
        }
        break;

    default:
        fprintf(stderr, "Unknown type %u\n", type);
        assert(false);
    }

    return value;
}

void GGUF_MetadataValueRelease(GGUF_MetadataType_t type, GGUF_MetadataValue_t value)
{
    switch (type)
    {
    case GGUF_METADATA_VALUE_TYPE_STRING:
        GGUF_StringRelease(value.str);
        break;

    case GGUF_METADATA_VALUE_TYPE_ARRAY:
        for (size_t i = 0; i < value.array.length; i++)
        {
            GGUF_MetadataValueRelease(value.array.type, value.array.array[i]);
        }
        free(value.array.array);
        break;

    case GGUF_METADATA_VALUE_TYPE_BOOL:
    case GGUF_METADATA_VALUE_TYPE_INT32:
    case GGUF_METADATA_VALUE_TYPE_UINT32:
    case GGUF_METADATA_VALUE_TYPE_UINT64:
    case GGUF_METADATA_VALUE_TYPE_FLOAT32:
        // Nothing to free...
        break;

    default:
        fprintf(stderr, "Cannot free type %u.\n", type);
        assert(false);
    }
}

void GGUF_MetadataValuePrint(GGUF_MetadataType_t type, GGUF_MetadataValue_t value)
{
    switch (type)
    {
    case GGUF_METADATA_VALUE_TYPE_STRING:
        printf("'%s'", value.str);
        break;
    case GGUF_METADATA_VALUE_TYPE_BOOL:
        printf("%s", value.bool_ ? "true" : "false");
        break;
    case GGUF_METADATA_VALUE_TYPE_INT32:
        printf("%i", value.int32);
        break;
    case GGUF_METADATA_VALUE_TYPE_UINT32:
        printf("%u", value.uint32);
        break;
    case GGUF_METADATA_VALUE_TYPE_UINT64:
        printf("%lu", value.uint64);
        break;
    case GGUF_METADATA_VALUE_TYPE_FLOAT32:
        printf("%f", value.float32);
        break;

    case GGUF_METADATA_VALUE_TYPE_ARRAY:
        printf("[");

        for (size_t i = 0; i < value.array.length; i++)
        {
            GGUF_MetadataValuePrint(value.array.type, value.array.array[i]);
            if (i != value.array.length - 1)
            {
                printf(", ");
            }
        }

        printf("]");
        break;

    default:
        fprintf(stderr, "Unknown type %u\n", type);
        assert(false);
    }
}

GGUF_Metadata_t GGUF_MetadataFromMemory(const uint8_t **data)
{
    GGUF_Metadata_t tmp;
    tmp.key = GGUF_StringFromMemory(data);
    assert(tmp.key);

    tmp.type = GGUF_MetadataTypeFromMemory(data);
    tmp.value = GGUF_MetadataValueFromMemory(tmp.type, data);
    return tmp;
}

void GGUF_MetadataRelease(GGUF_Metadata_t metadata)
{
    GGUF_MetadataValueRelease(metadata.type, metadata.value);
    GGUF_StringRelease(metadata.key);
}

void GGUF_MetadataPrint(GGUF_Metadata_t metadata)
{
    printf("'%s' = ", metadata.key);
    GGUF_MetadataValuePrint(metadata.type, metadata.value);
    printf("\n");
}

GGUF_TensorInfo_t GGUF_TensorInfoFromMemory(const uint8_t **data, const uint8_t *startOfFile)
{

    GGUF_TensorInfo_t info;

    info.name = GGUF_StringFromMemory(data);
    info.dimensionCount = GGUF_MetadataValueFromMemory(GGUF_METADATA_VALUE_TYPE_UINT32, data).uint32;

    size_t dimensionsSize = info.dimensionCount * sizeof(uint64_t);
    info.dimensions = malloc(dimensionsSize);
    assert(info.dimensions);
    memcpy(info.dimensions, *data, dimensionsSize);
    *data += dimensionsSize;

    info.type = GGUF_MetadataValueFromMemory(GGUF_METADATA_VALUE_TYPE_UINT32, data).uint32;
    uint64_t offset = GGUF_MetadataValueFromMemory(GGUF_METADATA_VALUE_TYPE_UINT64, data).uint64;

    switch (info.type)
    {
    case GGML_TYPE_F32:
        info.data.float32 = (const float *)(startOfFile + offset);
        break;
    case GGML_TYPE_F16:
        info.data.float16 = (const float16_t *)(startOfFile + offset);
        break;
    case GGML_TYPE_Q8_0:
        info.data.q8_0 = (const GGUF_Q8_0_t *)(startOfFile + offset);
        break;
    default:
        fprintf(stderr, "Unknown tensor data type %u\n", info.type);
        assert(false);
    }

    return info;
}

void GGUF_TensorInfoRelease(GGUF_TensorInfo_t info)
{
    GGUF_StringRelease(info.name);
    free(info.dimensions);
}

void GGUF_TensorInfoPrint(GGUF_TensorInfo_t info)
{

    printf("%s\t[", info.name);

    for (size_t i = 0; i < info.dimensionCount; i++)
    {
        printf("%lu", info.dimensions[i]);

        if (i < info.dimensionCount - 1)
        {
            printf(",");
        }
    }
    printf("]\ttype=%u\n", info.type);
}

const GGUF_Metadata_t *GGUF_MetadataFindByKey(const GGUF_Metadata_t *metadata, size_t count, const char *key)
{
    for (size_t i = 0; i < count; i++)
    {
        if (strcmp(metadata[i].key, key) == 0)
            return metadata + i;
    }
    return NULL;
}

bool GGUF_VerifyHeader(const uint8_t *data, size_t length)
{
    if (length < 24)
    {
        fprintf(stderr, "GGUF header too short.\n");
        return NULL;
    }
    const uint32_t magic = *((const uint32_t *)data);
    const uint32_t version = *((const uint32_t *)(data + 4));

    // Basic file format checks
    if (magic != 0x46554747)
    {
        fprintf(stderr, "Wrong magic number.\n");
        return false;
    }
    if (version != 3)
    {
        fprintf(stderr, "Unsupported version.\n");
        return false;
    }
    return true;
}

uint64_t GGUF_GetTensorCount(const uint8_t *data) { return *((const uint64_t *)(data + 8)); }

uint64_t GGUF_GetMetadataCount(const uint8_t *data) { return *((const uint64_t *)(data + 16)); }

const uint8_t *GGUF_SkipHeader(const uint8_t *data)
{
    return data + 24;
}

/******************************************************************************
 * Private Function Implementations
 ******************************************************************************/
