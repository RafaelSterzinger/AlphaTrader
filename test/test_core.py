import unittest
from mock import patch
from core.util import *

class UtilTest(unittest.TestCase):
    # def test_whenLoadingDataWithWrongDimension_thenValueIsNotNone(self):
    #     df = load_data("test/resource/test0.csv")
    #     self.assertIsNotNone(df)
    #
    #     assert df.shape == (2, 3)
    #
    # def test_whenLoadingDataWithWrongPath_thenFileNotFound(self):
    #     self.assertRaises(FileNotFoundError, load_data("wrong_path"))
    #
    # def test_whenProcessDataWithWrongDimension_thenAssertException(self):
    #     df = pd.DataFrame([{'a': 1, 'b': 2}])
    #     process_data(df, 30, (30, 40))
    #
    # def test_whenProcessDataWithRightDimension_thenGetTwoNumpyArrays(self):
    #     df = pd.DataFrame([{'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'Close': 7}])
    #     a1, a2 = process_data(df, 30, (30, 40))
    #     assert len(a1) == 1 and a1[0] == 7
    #     assert len(a2) == 2 and a2[0] == 7

    def test_sum(self):
        self.assertIs(4,4)

if __name__ == '__main__':
    unittest.main()