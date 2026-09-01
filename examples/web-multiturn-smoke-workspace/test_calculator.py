import unittest

from calculator import multiply


class CalculatorTests(unittest.TestCase):
    def test_multiplies_two_numbers(self):
        self.assertEqual(multiply(3, 4), 12)


if __name__ == "__main__":
    unittest.main()
