import unittest

from histogram.histogram import Histogram, HElementSet, operations
from histogram.utils import E
from histogram.executor import Parser, Evaluator, HistogramModel


class ParserTest(unittest.TestCase):

    def setUp(self):
        self.parser = Parser()

    def test(self):
        pass


class EvaluatorTest(unittest.TestCase):

    def setUp(self):

        # Input data
        data = ["e1", "e1", "e2", "e3", "e5", "e3", "e1", "e3", "e1", "e4"]

        # Transform the input data to histogram
        self.hist = Histogram(data)

        # Initialize Parser
        self.parser = Parser()

        # Define high level elements
        high_level_elements = {"E1": {"e1", "e2", "e3"}, "E2": {"e2", "e3", "e4"}}

        # Initialize Evaluator
        self.evaluator = Evaluator(operations, self.hist, high_level_elements=high_level_elements)

    def test_H_normalized(self):
        """Conversion input data into normalized histogram"""
        self.assertEqual(1, round(self.hist.sum(), 1), 1)
        self.assertEqual({"e2": 0.1, "e3": 0.3, "e1": 0.4, "e5": 0.1, "e4": 0.1}, self.hist.to_dict())

    def test_HE_from_H__low_level_element(self):
        """Retrieval histogram of low level element from data histogram"""
        HE = self.hist("e1")
        self.assertIsInstance(HE, HElementSet)
        self.assertDictEqual({"e1": 0.4}, HE.to_dict())
        self.assertAlmostEqual(0.4, HE.sum(), 2)

    def test_high_level_element(self):
        """Retrieval histogram of high level element from data histogram"""
        HE = self.hist("E1", {"E1": {"e1", "e2"}})
        self.assertIsInstance(HE, HElementSet)
        self.assertDictEqual({"e1": 0.4, "e2": 0.1}, HE.to_dict())
        self.assertAlmostEqual(0.5, HE.sum(), 2)

    def test_evaluate_Or__low_level_elements(self):
        """Evaluation of expression with low level elements and OR operation"""
        E1 = E("e1+e2+e3")     # compound elements
        E2 = E("e2+e3+e4")     # single element

        expression = self.parser.parse_string(E1.Or(E2).value)  # parse the expression
        HE_result = self.evaluator.eval(expression)             # evaluate the expression

        self.assertDictEqual({"e1": 0.4, "e2": 0.1, "e3": 0.3, "e4": 0.1}, HE_result.to_dict())
        self.assertAlmostEqual(0.9, HE_result.sum(), 2)

    def test_evaluate_Or__high_level_elements(self):
        """Evaluation of expression with high level element and OR operation"""

        E1 = E("E1")    # high level element
        E2 = E("e3")    # single element

        expression = self.parser.parse_string(E1.Or(E2).value)  # parse the expression
        HE_result = self.evaluator.eval(expression)             # evaluate the expression

        self.assertDictEqual({"e1": 0.4, "e2": 0.1, "e3": 0.3}, HE_result.to_dict())
        self.assertAlmostEqual(0.8, HE_result.sum(), 2)

    def test_evaluate_AndOr__low_level_elements(self):
        """Evaluation of expression with low level elements and OR operation"""
        E1 = E("e1+e2+e3")     # compound elements
        E2 = E("e2+e3+e4")     # single element

        expression = self.parser.parse_string((E1 + E2).value)  # parse the expression
        HE_result = self.evaluator.eval(expression)             # evaluate the expression

        self.assertDictEqual({"e1": 0.4, "e2": 0.1, "e3": 0.3, "e4": 0.1}, HE_result.to_dict())
        self.assertAlmostEqual(0.9, HE_result.sum(), 2)

    def test_evaluate_AndOr__high_level_elements(self):
        """Evaluation of expression with high level element and OR operation"""

        E1 = E("E1")    # high level element
        E2 = E("E2")    # high level element

        expression = self.parser.parse_string((E1 + E2).value)  # parse the expression
        HE_result = self.evaluator.eval(expression)             # evaluate the expression

        self.assertDictEqual({"e1": 0.4, "e2": 0.1, "e3": 0.3, "e4": 0.1}, HE_result.to_dict())
        self.assertAlmostEqual(0.9, HE_result.sum(), 2)


class Evaluator1DTest(unittest.TestCase):

    def setUp(self):
        # Input data
        data = [("ep1", "e1"), ("ep1", "e1"), ("ep1", "e2"),
                ("ep2", "e2"), ("ep2", "e2"), ("ep2", "e3"), ("ep2", "e3"),
                ("ep3", "e3"), ("ep3", "e3"), ("ep3", "e4")]

        # Transform the input data to histogram
        hist_model = HistogramModel(positioning="1d")
        self.hist = hist_model.transform(data)

        # Initialize Parser
        self.parser = Parser()

        # Define high level elements
        high_level_elements = {
            0: {"Ep1": {"ep1", "ep2"}, "Ep2": {"ep2", "ep3"}},
            1: {"E1": {"e1", "e2"}}
        }

        # Initialize Evaluator
        self.evaluator = Evaluator(operations, self.hist, high_level_elements=high_level_elements)

    def test_HE_from_H__all_position_element(self):
        HE = self.hist(("all", "e3"))
        self.assertIsInstance(HE, HElementSet)
        self.assertDictEqual({("ep2", "e3"): 0.2, ("ep3", "e3"): 0.2}, HE.to_dict())
        self.assertAlmostEqual(0.4, HE.sum(), 2)

    def test_evaluate__all_position_element(self):
        """Evaluation of expression with high level element and OR operation"""

        E1 = E("all, e3")

        expression = self.parser.parse_string(E1.value)         # parse the expression
        HE_result = self.evaluator.eval(expression)             # evaluate the expression

        self.assertDictEqual({("ep2", "e3"): 0.2, ("ep3", "e3"): 0.2}, HE_result.to_dict())
        self.assertAlmostEqual(0.4, HE_result.sum(), 2)

    def test_evaluate__high_level_elements(self):
        E1 = E("Ep1, E1")    # high level element

        expression = self.parser.parse_string(E1.value)         # parse the expression
        HE_result = self.evaluator.eval(expression)             # evaluate the expression

        self.assertDictEqual({("ep1", "e1"): 0.2, ("ep1", "e2"): 0.1, ("ep2", "e2"): 0.2}, HE_result.to_dict())
        self.assertAlmostEqual(0.5, HE_result.sum(), 2)

    def test_evaluate_AndOr__low_level_elements(self):
        """Evaluation of expression with low level elements and AndOr operation"""

        E1 = E("ep1, e1")
        E2 = E("ep2, e3")
        E3 = E("ep2, e3")

        query = (E1 + E2) + E3

        expression = self.parser.parse_string(query.value)      # parse the expression
        HE_result = self.evaluator.eval(expression)             # evaluate the expression

        self.assertDictEqual({("ep1", "e1"): 0.2, ("ep2", "e3"): 0.2}, HE_result.to_dict())
        self.assertAlmostEqual(0.4, HE_result.sum(), 2)

    def test_evaluate_AndOr__high_level_elements(self):
        """Evaluation of expression with high level element and AndOr operation"""

        E1 = E("Ep1, E1")    # high level element
        E2 = E("Ep2, e4")

        query = E1 + E2

        expression = self.parser.parse_string(query.value)      # parse the expression
        HE_result = self.evaluator.eval(expression)             # evaluate the expression

        self.assertDictEqual({("ep1", "e1"): 0.2, ("ep1", "e2"): 0.1, ("ep2", "e2"): 0.2, ("ep3", "e4"): 0.1},
                             HE_result.to_dict())
        self.assertAlmostEqual(0.6, HE_result.sum(), 2)


class HistogramTest(unittest.TestCase):

    def test_Union(self):

        data_1 = ["e1", "e1", "e2", "e3", "e5", "e3", "e1", "e3", "e1", "e4"]
        data_2 = ["e1", "e2", "e3", "e4", "e6"]

        hist_1 = Histogram(data_1, normalized=False)
        hist_2 = Histogram(data_2, normalized=False)

        hist = hist_1 + hist_2

        # hist.normalize()

        self.assertDictEqual({"e1": 5.0, "e2": 2.0, "e3": 4.0, "e4": 2.0, "e5": 1.0, "e6": 1.0}, hist.to_dict())
        self.assertAlmostEqual(15.0, hist.sum(), 2)

    def test_Union_normalized(self):
        pass

    def test_Intersection(self):

        data_1 = ["e1", "e1", "e2", "e3", "e5", "e3", "e1", "e3", "e1", "e4"]
        data_2 = ["e1", "e2", "e3", "e4", "e6"]

        hist_1 = Histogram(data_1, normalized=False)
        hist_2 = Histogram(data_2, normalized=False)

        hist = hist_1 * hist_2

        # hist.normalize()

        self.assertDictEqual({"e1": 1.0, "e2": 1.0, "e3": 1.0, "e4": 1.0}, hist.to_dict())
        self.assertAlmostEqual(4.0, hist.sum(), 2)

    def test_Intersection_normalized(self):
        pass


if __name__ == "__main__":
    unittest.main()
