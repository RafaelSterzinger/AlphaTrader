import pytest


class TestClass:

    def func(x):
        return x + 1

    def test_answer(self): assert TestClass.func(4) == 5
