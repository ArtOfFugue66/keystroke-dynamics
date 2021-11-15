import unittest
from features.utils import list_to_chunks_by_size, list_to_chunks_by_count


class TestUtilityFunctions(unittest.TestCase):
    def test_list_to_chunks_by_size_fcn(self):
        # Arrange
        test_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        test_chunk_size = 3
        expected = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]
        # Act
        actual = list_to_chunks_by_size(test_list, test_chunk_size)
        # Assert
        # Get items from generator
        actual_items = []
        for actual_item in actual:
            actual_items.append(actual_item)

        self.assertEqual(len(actual_items), len(expected))

        for i in range(0, len(expected)):
            self.assertListEqual(actual_items[i], expected[i])

    def test_list_to_chunks_by_count_fcn(self):
        # Arrange
        test_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        test_no_chunks = 4
        expected = [[1, 2, 3], [4, 5, 6],  [7, 8], [9, 10]]
        # Act
        actual = list_to_chunks_by_count(test_list, test_no_chunks)
        # Assert
        # Get items from generator
        actual_items = []
        for actual_item in actual:
            actual_items.append(actual_item)

        self.assertEqual(len(actual_items), len(expected))

        for i in range(0, len(expected)):
            self.assertListEqual(actual_items[i], expected[i])
