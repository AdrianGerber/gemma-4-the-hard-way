#include "unity.h"
#include "GGUF.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

void test_GGUF_StringFromMemory_valid_string(void);
void test_GGUF_StringFromMemory_empty_string(void);
void test_GGUF_StringRelease_null(void);
void test_GGUF_MetadataTypeFromMemory(void);
void test_GGUF_MetadataValueFromMemory_uint32(void);
void test_GGUF_MetadataValueFromMemory_bool(void);
void test_GGUF_MetadataValueFromMemory_string(void);
void test_GGUF_MetadataValueFromMemory_array(void);
void test_GGUF_MetadataValueRelease_string(void);
void test_GGUF_MetadataValueRelease_array(void);
void test_GGUF_MetadataFromMemory(void);
void test_GGUF_MetadataRelease(void);
void test_GGUF_TensorInfoFromMemory(void);
void test_GGUF_TensorInfoRelease(void);

void setUp(void)
{
}

void tearDown(void)
{
}

void test_GGUF_StringFromMemory_valid_string(void)
{
    // GGUF string: 8 bytes length (little endian) followed by characters
    // "Hello" is 5 characters. 5 in uint64_t LE: 05 00 00 00 00 00 00 00
    uint8_t data[] = {
        0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        'H', 'e', 'l', 'l', 'o'};
    const uint8_t *ptr = data;

    GGUF_String_t str = GGUF_StringFromMemory(&ptr);

    TEST_ASSERT_NOT_NULL(str);
    TEST_ASSERT_EQUAL_STRING("Hello", str);
    TEST_ASSERT_EQUAL_PTR(&data[13], ptr);

    GGUF_StringRelease(str);
}

void test_GGUF_StringFromMemory_empty_string(void)
{
    // Empty string: 8 bytes length 0
    uint8_t data[] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    const uint8_t *ptr = data;

    GGUF_String_t str = GGUF_StringFromMemory(&ptr);

    TEST_ASSERT_NOT_NULL(str);
    TEST_ASSERT_EQUAL_STRING("", str);
    TEST_ASSERT_EQUAL_PTR(&data[8], ptr);

    GGUF_StringRelease(str);
}

void test_GGUF_StringRelease_null(void)
{
    // Should not crash
    GGUF_StringRelease(NULL);
}

void test_GGUF_MetadataTypeFromMemory(void)
{
    // GGUF type is 32-bit uint
    uint8_t data[] = {0x04, 0x00, 0x00, 0x00}; // UINT32 (4)
    const uint8_t *ptr = data;

    GGUF_MetadataType_t type = GGUF_MetadataTypeFromMemory(&ptr);

    TEST_ASSERT_EQUAL(GGUF_METADATA_VALUE_TYPE_UINT32, type);
    TEST_ASSERT_EQUAL_PTR(&data[4], ptr);
}

void test_GGUF_MetadataValueFromMemory_uint32(void)
{
    uint8_t data[] = {0xEF, 0xBE, 0xAD, 0xDE}; // 0xDEADBEEF
    const uint8_t *ptr = data;

    GGUF_MetadataValue_t value = GGUF_MetadataValueFromMemory(GGUF_METADATA_VALUE_TYPE_UINT32, &ptr);

    TEST_ASSERT_EQUAL_UINT32(0xDEADBEEF, value.uint32);
    TEST_ASSERT_EQUAL_PTR(&data[4], ptr);
}

void test_GGUF_MetadataValueFromMemory_bool(void)
{
    uint8_t data[] = {0x01}; // true
    const uint8_t *ptr = data;

    GGUF_MetadataValue_t value = GGUF_MetadataValueFromMemory(GGUF_METADATA_VALUE_TYPE_BOOL, &ptr);

    TEST_ASSERT_TRUE(value.bool_);
    TEST_ASSERT_EQUAL_PTR(&data[1], ptr);
}

void test_GGUF_MetadataValueFromMemory_string(void)
{
    uint8_t data[] = {
        0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        'G', 'G', 'U'};
    const uint8_t *ptr = data;

    GGUF_MetadataValue_t value = GGUF_MetadataValueFromMemory(GGUF_METADATA_VALUE_TYPE_STRING, &ptr);

    TEST_ASSERT_EQUAL_STRING("GGU", value.str);
    TEST_ASSERT_EQUAL_PTR(&data[11], ptr);

    GGUF_MetadataValueRelease(GGUF_METADATA_VALUE_TYPE_STRING, value);
}

void test_GGUF_MetadataValueFromMemory_array(void)
{
    // Array of UINT32, 2 elements: [0x12345678, 0x9ABCDEF0]
    uint8_t data[] = {
        0x04, 0x00, 0x00, 0x00,                         // Element type: UINT32 (4)
        0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Length: 2
        0x78, 0x56, 0x34, 0x12,                         // Element 0
        0xF0, 0xDE, 0xBC, 0x9A                          // Element 1
    };
    const uint8_t *ptr = data;

    GGUF_MetadataValue_t value = GGUF_MetadataValueFromMemory(GGUF_METADATA_VALUE_TYPE_ARRAY, &ptr);

    TEST_ASSERT_EQUAL(GGUF_METADATA_VALUE_TYPE_UINT32, value.array.type);
    TEST_ASSERT_EQUAL_UINT64(2, value.array.length);
    TEST_ASSERT_NOT_NULL(value.array.array);
    TEST_ASSERT_EQUAL_UINT32(0x12345678, value.array.array[0].uint32);
    TEST_ASSERT_EQUAL_UINT32(0x9ABCDEF0, value.array.array[1].uint32);
    TEST_ASSERT_EQUAL_PTR(&data[20], ptr);

    GGUF_MetadataValueRelease(GGUF_METADATA_VALUE_TYPE_ARRAY, value);
}

void test_GGUF_MetadataValueRelease_string(void)
{
    // This is mostly to check it doesn't crash, and can be used with tools like Valgrind
    GGUF_MetadataValue_t value;
    value.str = malloc(10);
    strcpy(value.str, "test");

    GGUF_MetadataValueRelease(GGUF_METADATA_VALUE_TYPE_STRING, value);
}

void test_GGUF_MetadataValueRelease_array(void)
{
    GGUF_MetadataValue_t value;
    value.array.type = GGUF_METADATA_VALUE_TYPE_STRING;
    value.array.length = 1;
    value.array.array = malloc(sizeof(GGUF_MetadataValue_t));
    value.array.array[0].str = malloc(10);
    strcpy(value.array.array[0].str, "nested");

    GGUF_MetadataValueRelease(GGUF_METADATA_VALUE_TYPE_ARRAY, value);
}

void test_GGUF_MetadataFromMemory(void)
{
    // Key: "key" (3 chars), Type: UINT32 (4), Value: 0x12345678
    uint8_t data[] = {
        0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Key length
        'k', 'e', 'y',                                  // Key
        0x04, 0x00, 0x00, 0x00,                         // Type: UINT32
        0x78, 0x56, 0x34, 0x12                          // Value
    };
    const uint8_t *ptr = data;

    GGUF_Metadata_t metadata = GGUF_MetadataFromMemory(&ptr);

    TEST_ASSERT_EQUAL_STRING("key", metadata.key);
    TEST_ASSERT_EQUAL(GGUF_METADATA_VALUE_TYPE_UINT32, metadata.type);
    TEST_ASSERT_EQUAL_UINT32(0x12345678, metadata.value.uint32);
    TEST_ASSERT_EQUAL_PTR(&data[19], ptr);

    GGUF_MetadataRelease(metadata);
}

void test_GGUF_MetadataRelease(void)
{
    // Test that it doesn't crash on a complex metadata entry
    GGUF_Metadata_t metadata;
    metadata.key = malloc(4);
    assert(metadata.key);
    strcpy(metadata.key, "key");
    metadata.type = GGUF_METADATA_VALUE_TYPE_STRING;
    metadata.value.str = malloc(4);
    assert(metadata.value.str);
    strcpy(metadata.value.str, "val");

    GGUF_MetadataRelease(metadata);
}

void test_GGUF_TensorInfoFromMemory(void)
{
    // Tensor info:
    // Name: "weight" (6 chars)
    // Dimension count: 2
    // Dimensions: [128, 64]
    // Type: GGML_TYPE_F32 (0)
    // Offset: 1024
    uint8_t data[] = {
        0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Name length
        'w', 'e', 'i', 'g', 'h', 't',                   // Name
        0x02, 0x00, 0x00, 0x00,                         // Dimension count (UINT32)
        0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Dim 0: 128
        0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Dim 1: 64
        0x00, 0x00, 0x00, 0x00,                         // Type: GGML_TYPE_F32 (0) (UINT32)
        0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00  // Offset: 1024 (UINT64)
    };
    const uint8_t *ptr = data;
    uint8_t fakeFile[2048];
    memset(fakeFile, 0, sizeof(fakeFile));

    GGUF_TensorInfo_t info = GGUF_TensorInfoFromMemory(&ptr, fakeFile);

    TEST_ASSERT_EQUAL_STRING("weight", info.name);
    TEST_ASSERT_EQUAL_UINT32(2, info.dimensionCount);
    TEST_ASSERT_NOT_NULL(info.dimensions);
    TEST_ASSERT_EQUAL_UINT64(128, info.dimensions[0]);
    TEST_ASSERT_EQUAL_UINT64(64, info.dimensions[1]);
    TEST_ASSERT_EQUAL(GGML_TYPE_F32, info.type);
    TEST_ASSERT_EQUAL_PTR(&fakeFile[1024], info.data.float32);
    TEST_ASSERT_EQUAL_PTR(&data[sizeof(data)], ptr);

    GGUF_TensorInfoRelease(info);
}

void test_GGUF_TensorInfoRelease(void)
{
    GGUF_TensorInfo_t info;
    info.name = malloc(10);
    strcpy(info.name, "tensor");
    info.dimensionCount = 1;
    info.dimensions = malloc(sizeof(uint64_t));
    info.dimensions[0] = 10;
    info.type = GGML_TYPE_F32;
    // data is a pointer to something else, not owned by info

    GGUF_TensorInfoRelease(info);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_GGUF_StringFromMemory_valid_string);
    RUN_TEST(test_GGUF_StringFromMemory_empty_string);
    RUN_TEST(test_GGUF_StringRelease_null);
    RUN_TEST(test_GGUF_MetadataTypeFromMemory);
    RUN_TEST(test_GGUF_MetadataValueFromMemory_uint32);
    RUN_TEST(test_GGUF_MetadataValueFromMemory_bool);
    RUN_TEST(test_GGUF_MetadataValueFromMemory_string);
    RUN_TEST(test_GGUF_MetadataValueFromMemory_array);
    RUN_TEST(test_GGUF_MetadataValueRelease_string);
    RUN_TEST(test_GGUF_MetadataValueRelease_array);
    RUN_TEST(test_GGUF_MetadataFromMemory);
    RUN_TEST(test_GGUF_MetadataRelease);
    RUN_TEST(test_GGUF_TensorInfoFromMemory);
    RUN_TEST(test_GGUF_TensorInfoRelease);
    return UNITY_END();
}
