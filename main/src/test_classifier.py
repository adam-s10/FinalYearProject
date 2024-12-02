import logging
import os
import sys
import unittest
from classifier import FileManager


class TestFileManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logger.info('TestFileManager: setUpClass starting, creating files')
        with open('D:/PycharmProjects/FinalYearProject/main/src/test_file_read.txt', 'w') as read_file_test:
            read_file_test.write('This should be readable...')
        with open('D:/PycharmProjects/FinalYearProject/main/src/test_file_append.txt', 'a') as append_file_test:
            append_file_test.write('This should be ')
        logger.info('TestFileManager: setUpClass finished, files created')

    def test_string_representation(self):
        file = FileManager('test_string_representation', 'w')
        expected_result = ('FileManager for test_string_representation: \nWorking in mode --> w\nWith root --> '
                           'D:\\PycharmProjects\\FinalYearProject\\main\\src')

        self.assertEqual(repr(file), expected_result,
                         'String representation of class was not returned as expected!')

    def test_context_manager_suppresses_exception(self):
        # Test that an exception is suppressed by the context manager
        with FileManager('test_context_manager_suppresses_exception', 'w') as file:
            file.add(90)  # There is no 'add' method defined; AttributeError will be raised

    def test_file_read(self):
        with FileManager('test_file_read', 'r') as file:
            self.assertEqual(file.read(), 'This should be readable...', 'File was not read from correctly!')

    def test_file_append(self):
        with FileManager('test_file_append', 'a') as file:
            file.write('appendable...')
        with open('D:/PycharmProjects/FinalYearProject/main/src/test_file_append.txt', 'r') as read_file:
            self.assertEqual('This should be appendable...', read_file.read(),
                             'File was not appended to correctly!')

    def test_file_write(self):
        test_var = 'This is a test!'
        with FileManager('test_file_write', 'w') as file:
            file.write(test_var)
        with open('D:/PycharmProjects/FinalYearProject/main/src/test_file_write.txt', 'r') as read_file:
            self.assertEqual(test_var, read_file.read(), 'File was not written into correctly!')

    @classmethod
    def tearDownClass(cls):
        logger.info('TestFileManager: tearDownClass starting, deleting files')
        os.remove('D:/PycharmProjects/FinalYearProject/main/src/test_file_read.txt')
        os.remove('D:/PycharmProjects/FinalYearProject/main/src/test_file_append.txt')
        os.remove('D:/PycharmProjects/FinalYearProject/main/src/test_context_manager_suppresses_exception.txt')
        os.remove('D:/PycharmProjects/FinalYearProject/main/src/test_file_write.txt')
        logger.info('TestFileManager: tearDownClass finished, files deleted')


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

if __name__ == '__main__':
    unittest.main()
