import unittest

from calculator import add, multiply


class CalculatorOperationTests(unittest.TestCase):
    def test_adds_two_integers(self) -> None:
        self.assertEqual(add(3, 4), 7)

    def test_multiplies_positive_integers(self) -> None:
        self.assertEqual(multiply(3, 4), 12)

    def test_multiplies_by_zero(self) -> None:
        self.assertEqual(multiply(8, 0), 0)


if __name__ == "__main__":
    unittest.main()
