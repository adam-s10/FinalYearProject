import unittest
from classifier import FileManager


class TestClassifier(unittest.TestCase):
    # def test_something(self):
    #     self.assertEqual(True, False)  # add assertion here

    def test_string_representation(self):
        file = FileManager('test_string_representation', 'w')
        expected_result = ('FileManager for test_string_representation: \nWorking in mode --> w\nWith root --> '
                           'D:\\PycharmProjects\\FinalYearProject\\main\\src')

        self.assertEqual(file.__repr__(), expected_result)

    def test_context_manager_suppresses_exception(self):
        # Test that an exception is suppressed by the context manager
        with FileManager('test_context_manager_suppresses_exception', 'w') as file:
            file.add(90)  # There is no 'add' method defined; AttributeError will be raised


if __name__ == '__main__':
    unittest.main()
