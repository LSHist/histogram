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
    oneOf
)


class Parser:
    """Parser for element expressions/queries"""
    def __init__(self, parser_definition=None):
        self._expr = parser_definition or self._create_parser()

    def parse_string(self, expression, output_type="postfix"):
        if output_type == "postfix":
            if hasattr(self, "_postfix") and isinstance(self._postfix, list):
                self._postfix[:] = []
            else:
                self._postfix = []
            self._expr.parseString(expression)
            return self._postfix
        elif output_type == "infix":
            return self._expr.parseString(expression)

    def parse_list(self, element_list):
        pass

    def _create_parser(self):
        """
        Parser
        ------
        op      :: '+' | '-' | '/' | '&' | '|' | '#|'
        element :: ['+' | '-'] ['a'..'z']['A'..'Z']+
        term    :: element | '(' expr ')'
        expr    :: term [ op term ]*
        """
        element = Word("+-" + alphas, alphanums)
        op = oneOf("+ - * / & | #| #/")
        lpar, rpar = map(Suppress, "()")
        expr = Forward()
        term = (op[...] + (element.setParseAction(self._push_first) | Group(lpar + expr + rpar)))\
            .setParseAction(self._push_unary_minus)
        expr <<= term + (op + term).setParseAction(self._push_first)[...]
        return expr

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

        if input_type == "postfix":
            return self._postfix_evaluate(expr, data_histogram if data_histogram else self._H)
        else:
            raise NotImplemented("Not implemented yet.")

    def _postfix_evaluate(self, expression, histogram):

        op, num_args = expression.pop(), 0

        if isinstance(op, tuple):
            op, num_args = op
        if op == "unary -":
            return -self._postfix_evaluate(expression, histogram)
        if op in self._O.keys():
            op2 = self._postfix_evaluate(expression, histogram)
            op1 = self._postfix_evaluate(expression, histogram)
            return self._O[op](op1, op2)
        else:
            if op in self._extendedE:
                return histogram(op, self._extendedE[op])
            return histogram(op)
