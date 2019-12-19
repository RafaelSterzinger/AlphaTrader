import unittest
from core.util import *


class UtilTest(unittest.TestCase):

    def test_whenLoadingDataWithWrongPath_thenFileNotFound(self):
        self.assertRaises(FileNotFoundError, load_data("wrong_path"))

    def test_whenProcessDataWithWrongDimension_thenAssertError(self):
        df = pd.DataFrame([{'a': 1, 'b': 2}])
        try:
            process_data(df, 30, (30, 40))
            self.fail()
        except AssertionError as e:
            print("OK")
        except BaseException as e:
            self.fail()

    def test_whenProcessDataWithRightDimension_thenGetTwoNumpyArrays(self):
        df = pd.DataFrame([{'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'Close': 7}])
        a1, a2 = process_data(df, 30, (30, 40))
        self.assertEqual(1, len(a1))
        self.assertEqual(1, len(a2))
        self.assertEqual(7, a1[0])
        self.assertEqual(7, a2[0])

    def test_whenStandardScaling_thenAllValuesBetween0and1(self):
        df = [[1], [2], [3], [4], [5]]
        df = standard_scale(df)
        for i in df:
            self.assertGreaterEqual(i, -2)
            self.assertLessEqual(i, 2)

    def test_whenMinMaxScaling_thenAllValuesBetween0and1(self):
        df = [[1], [2], [3], [4], [5]]
        df = min_max_scale(df)
        for i in df:
            self.assertGreater(i, 0)
            self.assertLessEqual(i, 1)

    def test_whenSigmoidScaling_thenAllValuesBetween0and1(self):
        df = [[1], [2], [3], [4], [5]]
        df = sigmoid_scale(df)
        for i in df:
            self.assertGreater(i, 0)
            self.assertLessEqual(i, 1)


if __name__ == '__main__':
    unittest.main()
