import unittest
from lshist.histogram import Histogram, HElementSet, operations
from lshist.utils import E
from lshist.executor import Parser, Evaluator


class ParserTest(unittest.TestCase):

    def setUp(self):
        self.parser = Parser()

    def test(self):
        pass


class EvaluatorTest(unittest.TestCase):

    def setUp(self):

        # Input data
        data = ["e1", "e1", "e2", "e3", "e5", "e3", "e1", "e3", "e1", "e4"]

        # TODO: [(e11, e12, e13), (e21, e22, e23), ...]

        # Transform the input data to histogram
        self.hist = Histogram(data)

        # Initialize Parser
        self.parser = Parser()

        # Define high level elements
        high_level_elements = {"E1": {"e1", "e2"}}

        # Initialize Evaluator
        self.evaluator = Evaluator(self.hist, operations, high_level_elements=high_level_elements)

    def test_H_normalized(self):
        """Conversion input data into normalized histogram"""
        self.assertEqual(1, round(self.hist.sum(), 1), 1)
        self.assertEqual({"e2": 0.1, "e3": 0.3, "e1": 0.4, "e5": 0.1, "e4": 0.1}, self.hist.to_dict())

    def test_HE_from_H__low_level_element(self):
        """Retrieval histogram of low level element from data histogram"""
        HE = self.hist("e1")
        self.assertIsInstance(HE, HElementSet)
        self.assertDictEqual({"e1": 0.4}, HE.to_dict())
        self.assertEqual(0.4, round(HE.sum(), 2))

    def test_high_level_element(self):
        """Retrieval histogram of high level element from data histogram"""
        HE = self.hist("E1", {"e1", "e2"})
        self.assertIsInstance(HE, HElementSet)
        self.assertDictEqual({"e1": 0.4, "e2": 0.1}, HE.to_dict())
        self.assertEqual(0.5, round(HE.sum(), 2))

    def test_evaluate_Or__low_level_elements(self):
        """Evaluation of expression with low level elements and OR operation"""
        E1 = E("e1+e2")     # compound elements
        E2 = E("e3")        # single element

        expression = self.parser.parse_string(E1.Or(E2).value)  # parse the expression
        HE_result = self.evaluator.eval(expression)             # evaluate the expression

        self.assertDictEqual({"e1": 0.4, "e2": 0.1, "e3": 0.3}, HE_result.to_dict())
        self.assertEqual(0.8, round(HE_result.sum(), 2))

    def test_evaluate_Or__high_level_elements(self):
        """Evaluation of expression with high level element and OR operation"""

        E1 = E("E1")    # high level element
        E2 = E("e3")    # single element

        expression = self.parser.parse_string(E1.Or(E2).value)  # parse the expression
        HE_result = self.evaluator.eval(expression)             # evaluate the expression

        self.assertDictEqual({"e1": 0.4, "e2": 0.1, "e3": 0.3}, HE_result.to_dict())
        self.assertEqual(0.8, round(HE_result.sum(), 2))


if __name__ == "__main__":
    unittest.main()
