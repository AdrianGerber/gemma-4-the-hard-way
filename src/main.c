#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "GGUF.h"
#include <string.h>
#include <assert.h>
#include "Tokenizer.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////
typedef struct
{
    GGUF_Metadata_t *metadata;
    size_t metadataCount;
    GGUF_TensorInfo_t *tensorInfo;
    size_t tensorInfoCount;
} Model_t;

Model_t *ParseModelFromGGUF(const uint8_t *data, size_t length);
void ReleaseModel(Model_t *model);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int main(void)
{
    const char *filename = "gemma-4-e2b-it-Q8_0.gguf";
    int f = open(filename, O_RDONLY);
    if (f == -1)
    {
        perror("open failed");
        exit(1);
    }

    struct stat fileStat;
    if (fstat(f, &fileStat) != 0)
    {
        perror("stat failed");
        close(f);
        exit(1);
    }

    const uint8_t *data = mmap(NULL, (size_t)fileStat.st_size, PROT_READ, MAP_SHARED, f, 0);
    if (data == MAP_FAILED)
    {
        perror("mmap failed");
        close(f);
        exit(1);
    }

    Model_t *model = ParseModelFromGGUF(data, (size_t)fileStat.st_size);
    if (model)
    {

        printf("\n\n");

        Tokenizer_t tokenizer = Tokenizer_Init(model->metadata, model->metadataCount);

        const char *test =
            "<bos>\n"
            "<|turn>user\n"
            "Hello world<turn|>\n"
            "<|turn>model\n";

        printf("Encoding Text '%s'\n", test);
        TokenizerEncoded_t tokenIds = Tokenizer_Encode(tokenizer, test);
        printf("Tokenized: ");
        Tokenizer_DecodeToStdOut(tokenizer, tokenIds, true);
        printf("\n");

        Tokenizer_Release(tokenizer);

        ReleaseModel(model);
        model = NULL;
    }

    munmap((void *)data, (size_t)fileStat.st_size);
    close(f);
    return 0;
}

Model_t *ParseModelFromGGUF(const uint8_t *data, size_t length)
{
    if (length < 24)
    {
        fprintf(stderr, "GGUF header too short.\n");
        return NULL;
    }

    // Parse basic header information
    const uint32_t magic = *((const uint32_t *)data);
    const uint32_t version = *((const uint32_t *)(data + 4));
    const uint64_t tensorCount = *((const uint64_t *)(data + 8));
    const uint64_t metadataCount = *((const uint64_t *)(data + 16));
    const uint8_t *metadata = data + 24;
    printf("GGUF Header:\n");
    printf("magic = %08x\n", magic);
    printf("version = %u\n", version);
    printf("number of tensors = %lu\n", tensorCount);
    printf("number of metadata values = %lu\n", metadataCount);

    // Basic file format checks
    if (magic != 0x46554747)
    {
        fprintf(stderr, "Wrong magic number.\n");
        return NULL;
    }
    if (version != 3)
    {
        fprintf(stderr, "Unsupported version.\n");
        return NULL;
    }

    // Parse metadata
    printf("\nMetadata:\n");
    Model_t *model = malloc(sizeof(Model_t));
    assert(model);
    model->metadataCount = metadataCount;
    model->metadata = malloc(metadataCount * sizeof(GGUF_Metadata_t));
    assert(model->metadata);
    for (size_t i = 0; i < metadataCount; i++)
    {
        model->metadata[i] = GGUF_MetadataFromMemory(&metadata);

        if (model->metadata[i].type == GGUF_METADATA_VALUE_TYPE_ARRAY && model->metadata[i].value.array.length > 10)
        {
            printf("'%s' = [...]\n", model->metadata[i].key);
        }
        else if (model->metadata[i].type == GGUF_METADATA_VALUE_TYPE_STRING && strlen(model->metadata[i].value.str) > 1024)
        {
            printf("'%s' = '...'\n", model->metadata[i].key);
        }
        else
        {
            GGUF_MetadataPrint(model->metadata[i]);
        }
    }

    // Parse tensor infos
    printf("\nTensors:\n");
    model->tensorInfoCount = tensorCount;
    model->tensorInfo = malloc(tensorCount * sizeof(GGUF_TensorInfo_t));
    assert(model->tensorInfo);
    for (size_t i = 0; i < tensorCount; i++)
    {
        model->tensorInfo[i] = GGUF_TensorInfoFromMemory(&metadata, data);
        GGUF_TensorInfoPrint(model->tensorInfo[i]);
    }

    // TODO: memory length safety

    return model;
}

void ReleaseModel(Model_t *model)
{
    if (model->metadata)
    {
        for (size_t i = 0; i < model->metadataCount; i++)
        {
            GGUF_MetadataRelease(model->metadata[i]);
        }
        free(model->metadata);
    }

    if (model->tensorInfo)
    {
        for (size_t i = 0; i < model->tensorInfoCount; i++)
        {
            GGUF_TensorInfoRelease(model->tensorInfo[i]);
        }
        free(model->tensorInfo);
    }

    free(model);
}
