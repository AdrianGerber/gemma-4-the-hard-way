#include "unity.h"
#include "file2.h"

void test_add_integers_positive_numbers(void);
void test_add_integers_negative_numbers(void);

void setUp(void)
{
}
void tearDown(void)
{
}

void test_add_integers_positive_numbers(void)
{
    TEST_ASSERT_EQUAL_INT(5, add_integers(2, 3));
}

void test_add_integers_negative_numbers(void)
{
    TEST_ASSERT_EQUAL_INT(-1, add_integers(2, -3));
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_add_integers_positive_numbers);
    RUN_TEST(test_add_integers_negative_numbers);
    return UNITY_END();
}
