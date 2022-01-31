from himpy.executor import Parser, Evaluator
from himpy.histogram import Histogram


class SearchEngine:
    """Simple search engine based on iterating over entire datasets."""

    def __init__(self, hists, parser: Parser, evaluator: Evaluator):
        self._hists = hists
        self._parser = parser
        self._evaluator = evaluator

    def retrieve(self, query, topN=10, lastN=None, threshold=0.001):
        img_rank = list()
        if hasattr(query, "value") and isinstance(query.value, str):
            """Searching by expression"""
            expr = self._parser.parse_string(query.value)
            HEs = [(img_id, self._evaluator.eval(expr, hist)) for img_id, hist in self._hists]
            img_rank = sorted(
                [(img_id, HE.sum()) for img_id, HE in HEs if HE.sum() > threshold],
                key=lambda x: -x[1]
            )
        elif isinstance(query, Histogram):
            """Searching by data histogram"""
            img_rank = sorted(
                [(image_id, (query * hist).sum()) for image_id, hist in self._hists],
                key=lambda x: -x[1]
            )
        if isinstance(lastN, int):
            return img_rank[:topN], img_rank[-lastN:]

        return img_rank[:topN]

