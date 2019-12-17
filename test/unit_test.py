import unittest
from core.util import *


class UtilTest(unittest.TestCase):

    def whenLoadingDataWithWrongDimension_thenValueIsNotNone(self):
        df = load_data("test/resource/test0.csv")
        assert df is not None
        assert df.shape == (2, 3)

    def whenLoadingDataWithWrongPath_thenFileNotFound(self):
        self.assertRaises(FileNotFoundError, load_data("wrong_path"))

    def whenProcessDataWithWrongDimension_thenAssertException(self):
        df = pd.DataFrame([{'a': 1, 'b': 2}])
        process_data(df, 30, (30, 40))

    def whenProcessDataWithRightDimension_thenGetTwoNumpyArrays(self):
        df = pd.DataFrame([{'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'Close': 7}])
        a1, a2 = process_data(df, 30, (30, 40))
        assert len(a1) == 1 and a1[0] == 7
        assert len(a2) == 2 and a2[0] == 7

if __name__ == '__main__':
    unittest.main()
