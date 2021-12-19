"""
Parser and Evaluator for Histogram Model
"""

from pyparsing import (
    Word,
    Group,
    Forward,
    alphas,
    alphanums,
    Suppress,
    oneOf,
    delimitedList, ParseResults
)

from .histogram import (
    operations,
    Histogram,
    Histogram1D,
    Histogram2D
)


class Parser:
    """Parser for element expressions/queries"""
    def __init__(self, parser_definition=None):
        self._expr = parser_definition or self._create_parser()

    def parse_string(self, expression, output_type="postfix", copy_expression=True):
        if hasattr(self, "_postfix") and isinstance(self._postfix, list):
            self._postfix[:] = []
        else:
            self._postfix = []
        if output_type == "postfix":
            self._expr.parseString(expression)
            return self._postfix.copy() if copy_expression is True else self._postfix
        elif output_type == "infix":
            return self._expr.parseString(expression).copy() if copy_expression is True \
                else self._expr.parseString(expression)

    def parse_set(self, element_list, sep="+"):
        return set(element_list.strip(" ()").split(sep))

    def _create_parser(self):

        """
        Parser
        ------
        op      :: '+' | '-' | '/' | '&' | '|' | '#|'
        element :: ['+' | '-'] ['a'..'z']['A'..'Z']+
        term    :: element | '(' expr ')'
        expr    :: term [ op term ]*
        """
        lpar, rpar = map(Suppress, "()")
        element = Word("+-" + alphas, alphanums + "_")
        op = oneOf("+ - * / & | #| #/")
        expr = Forward()
        complex_element = element | Group(lpar + delimitedList(element) + rpar)
        term = (op[...] + (complex_element.setParseAction(self._push_complex_element)
                           | Group(lpar + expr + rpar)))\
            .setParseAction(self._push_unary_minus)
        expr <<= term + (op + term).setParseAction(self._push_first)[...]
        return expr

    def _push_complex_element(self, tokens):
        """Postfix notation for binary operations"""
        if isinstance(tokens[0], ParseResults):
            if len(tokens[0]) == 1:
                self._postfix.append(tokens[0][0])
            else:
                self._postfix.append(tuple(tokens[0]))
        else:
            self._postfix.append(tokens[0])

    def _push_first(self, tokens):
        """Postfix notation for binary operations"""
        self._postfix.append(tokens[0])

    def _push_unary_minus(self, tokens):
        """Postfix notation for unary operations"""
        for t in tokens:
            if t == "-":
                self._postfix.append("unary -")
            else:
                break


class Evaluator:
    """Evaluator for parsed element expressions/queries"""
    def __init__(self, operators, histogram=None, high_level_elements=None):
        self._H = histogram
        self._O = operators
        self._extendedE = high_level_elements or dict()

    def eval(self, expression, data_histogram=None, input_type="postfix", copy_expression=True):

        if copy_expression:
            expr = expression.copy()
        else:
            expr = expression

        hist = data_histogram if data_histogram is not None else self._H

        if input_type == "postfix":
            return self._postfix_evaluate(expr, hist)
        else:
            raise NotImplemented("Not implemented yet.")

    def _postfix_evaluate(self, expression, histogram):

        op, num_args = expression.pop(), 0

        if op == "unary -":
            return -self._postfix_evaluate(expression, histogram)
        if op in self._O.keys():
            op2 = self._postfix_evaluate(expression, histogram)
            op1 = self._postfix_evaluate(expression, histogram)
            return self._O[op](op1, op2)
        else:
            return histogram(op, self._extendedE)


class HistogramModel:

    def __init__(self, U=None, positioning=False, U_positions=None):

        self.U = U
        self.positioning = positioning
        self.U_positions = U_positions
        self.evaluator = None
        self.parser = None
        self.operations = operations

    def execute(self, query, histogram):

        if not hasattr(self, "parser") or not self.parser:
            self.parser = Parser()

        if not hasattr(self, "evaluator") or not self.evaluator:
            self.evaluator = Evaluator(operations)

        query_array = self.parser.parse_string(query)
        return self.evaluator.eval(query_array, histogram)

    def transform(self, data):

        if not data:
            raise Exception("No data to transform.")

        # if not hasattr(self, "U") or not self.U:
        #     raise Exception("There are no elements to transform the data.")
        #
        # if self.positioning and (not hasattr(self, "U_positions") or not self.U_positions):
        #     raise Exception("There are no position elements to transform the data.")

        if self.positioning is None:
            return self._transform2histogram(data)
        elif self.positioning == "1d":
            return self._transform2histogram1D(data)
        elif self.positioning == "2d":
            return self._transform2histogram2D(data)
        else:
            raise Exception("Wrong positioning argument.")

    def _transform2histogram(self, data):
        return Histogram(data)

    def _transform2histogram1D(self, data):

        if not isinstance(data, (list, tuple)) or not isinstance(data[0], (list, tuple)) or \
                len(data[0]) <= 1:  # TODO: add support np.ndarray
            raise Exception("Wrong data format.")

        return Histogram1D(data)

    def _transform2histogram2D(self, data):

        if not isinstance(data, (list, tuple)) or not isinstance(data[0], (list, tuple)) or \
                len(data[0]) <= 1:  # TODO: add support np.ndarray
            raise Exception("Wrong data format.")

        return Histogram2D(data)
