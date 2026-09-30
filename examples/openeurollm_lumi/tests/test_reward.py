import unittest

from reward import compute_score


class RewardTests(unittest.TestCase):
    def test_integer(self):
        self.assertEqual(compute_score("math", r"<think>work</think>\boxed{31}", "31"), 1.0)
        self.assertEqual(compute_score("math", r"\boxed{30}", "31"), 0.0)

    def test_rational(self):
        self.assertEqual(compute_score("math", r"\boxed{\frac{6}{8}}", "3/4"), 1.0)
        self.assertEqual(compute_score("math", r"\boxed{0.75}", "3/4"), 1.0)

    def test_requires_complete_box(self):
        self.assertEqual(compute_score("math", "31", "31"), 0.0)
        self.assertEqual(compute_score("math", r"\boxed{31", "31"), 0.0)
        self.assertEqual(compute_score("math", r"\boxed{1/0}", "31"), 0.0)


if __name__ == "__main__":
    unittest.main()
