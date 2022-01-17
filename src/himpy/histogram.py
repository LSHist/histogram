"""
Histogram Data Structures
"""


class Element:
    pass


class HElement:
    """
    Low Level Histogram Element

    This element must corresponds to one of elements from the universal set.

    Note: The universal set is one from which data is made up. Think of it as
    a dictionary of terms.

    Parameters
    ----------
    key         an element id
    value       a value of the element
    properties  additional parameters that can be used in evaluation phase,
    e.g. mean, var

    """

    def __init__(self, key, value, properties=None):
        self._key = key
        self._value = value
        self._properties = properties or list()

    @property
    def key(self):
        return self._key

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def __hash__(self):
        return hash(self._key)

    def __eq__(self, other):
        return isinstance(other, HElement) and self._key == other.key

    def __ne__(self, other):
        return isinstance(other, HElement) and self._key != other.key


class HElementCollection:
    """Base class for a histogram of elements (HE)"""

    def add(self, he):
        """Add the new element to the histogram"""
        raise NotImplementedError

    def discard(self, he):
        """Discard the element from the histogram"""
        raise NotImplementedError


class HElementSet(HElementCollection):
    """
    Histogram of elements implemented as a set

    Parameters
    ----------
    _HE     set of elements of type HElement
    """

    def __init__(self, HE=None):
        super(HElementSet, self).__init__()
        self._HE = HE or set()

    def add(self, he):
        if isinstance(he, HElement):
            self._HE.add(he)
        else:
            raise Exception("Argument is not HElement")

    def discard(self, he):
        if isinstance(he, HElement):
            self._HE.discard(he)
        else:
            raise Exception("Argument is not HElement")

    def union(self, other):
        if isinstance(other, HElementSet):
            return self._HE.union(other)
        raise Exception("Argument is not HElement")

    def intersection(self, other):
        if isinstance(other, HElementSet):
            return self._HE.intersection(other)
        raise Exception("Argument is not HElement")

    def difference(self, other):
        if isinstance(other, HElementSet):
            return self._HE.difference(other)
        raise Exception("Argument is not HElement")

    def sum(self):
        """Sum of all elements of histogram"""
        return sum(el.value for el in self._HE)

    def prod(self):
        """Product of all elements of histogram"""
        result = 1
        for el in self._HE:
            result = result * el.value
        return result

    def elements(self):
        """List of ids of non-zero histogram elements"""
        if hasattr(self, "_HE") and isinstance(self._HE, set):
            return [el.key for el in self._HE]
        raise Exception("There are no elements.")

    def values(self):
        """List of values of non-zero histogram elements"""
        if hasattr(self, "_HE") and isinstance(self._HE, set):
            return [el.value for el in self._HE]
        raise Exception("There are no elements.")

    def to_dict(self):
        """Dictionary of non-zero histogram elements: {id:value}"""
        if hasattr(self, "_HE") and isinstance(self._HE, set):
            return {el.key: el.value for el in self._HE}
        raise Exception("There are no elements.")

    def __contains__(self, item):
        if isinstance(item, HElement):
            return item in self._HE
        elif isinstance(item, str):
            return HElement(item, 0) in self._HE
        return False

    def __len__(self):
        return len(self._HE)

    def __iter__(self):
        return self._HE.__iter__()


class Histogram:
    """Data Histogram"""

    def __init__(self, data, normalized=True, size=None, transform_func=None):

        if not data:
            self._HE = dict()
            return

        if transform_func:
            self._HE = transform_func(data)
        else:
            self._HE = Histogram.transform(data)
        self._size = size or sum(el.value for el in self._HE.values())
        self._normalized = False
        if normalized:
            self._normalize()
            self._normalized = True

    def sum(self):
        if not hasattr(self, "_HE") or not isinstance(self._HE, dict):
            raise Exception("There are no elements.")
        return sum(self._HE[key].value for key in self._HE)

    # TODO: count histogram elements
    # TODO: count elements

    def elements(self):
        if hasattr(self, "_HE") and isinstance(self._HE, dict):
            return list(self._HE.keys())
        raise Exception("There are no elements.")

    def hist_elements(self):
        if hasattr(self, "_HE") and isinstance(self._HE, dict):
            return self._HE
        raise Exception("There are no elements.")

    def add(self, element):

        if isinstance(element, HElement):
            if element.key not in self._HE:
                self._HE[element.key] = HElement(element.key, 0)
            self._HE[element.key].value += element.value
            self._size += element.value

    def to_dict(self):
        """Dictionary of non-zero histogram elements: {id:value}"""
        if hasattr(self, "_HE") and isinstance(self._HE, dict):
            return {key: value.value for key, value in self._HE.items()}
        raise Exception("There are no elements.")

    def normalize(self, size=None):
        if size:
            self._size = size
        self._normalize()

    def _normalize(self):
        """Normalize data histogram to 1"""
        for key in self._HE:
            self._HE[key].value = float(self._HE[key].value) / self._size
        self._normalized = True

    def __call__(self, element, composition=None):
        """
        Create a histogram of elements

        Parameters
        ----------
        element         a low- or high-level element
        composition     used for a high-level element to define a set of low-level elements

        Returns
        -------
        histogram of elements (HE) -> HElementSet

        """
        if element in self:
            return HElementSet(HE={self[element]})
        elif element and composition and element in composition and isinstance(composition[element], set):
            return HElementSet(HE={self[el] for el in composition[element] if el in self})
        return HElementSet()

    def __setitem__(self, key, value):
        self._HE[key] = value

    def __getitem__(self, item):
        return self._HE[item]

    def __contains__(self, item):
        return item in self._HE

    def __len__(self):
        return hasattr(self, "_HE") and len(self._HE)

    def __add__(self, other):
        return hist_operations["+"].compute(self, other)

    def __mul__(self, other):
        return hist_operations["*"].compute(self, other)

    def __iter__(self):
        return self._HE.items().__iter__()

    @staticmethod
    def transform(data):
        """
        Convert data to histogram

        Parameters
        ----------
        data            a data composed from elements of the universal set

        Returns
        -------
        dictionary      {element id : value}

        """
        histogram = dict()
        if isinstance(data, list):
            for el in data:
                if el not in histogram:
                    histogram[el] = HElement(el, 0.0)
                histogram[el].value += 1.0
        return histogram


class Histogram1D(Histogram):
    """Data Histogram with 1D positioning"""

    def __call__(self, element, composition=None):
        """
        Create a histogram of elements

        Parameters
        ----------
        element         a low- or high-level element
        composition     used for a high-level element to define a set of low-level elements

        Returns
        -------
        histogram of elements (HE) -> HElementSet

        """

        # element
        element_ndim = len(element) if isinstance(element, tuple) else 1

        Es = dict()
        has_compound = False
        if element_ndim == 1:
            Es = {element}
            if composition is not None and element in composition:
                Es = composition[element]
                has_compound = True
        elif element_ndim > 1:
            for i in range(element_ndim):
                Es[i] = {element[i]}
                if composition is not None and i in composition and element[i] in composition[i]:
                    Es[i] = composition[i][element[i]]
                    has_compound = True

        if not has_compound and element in self:
            return HElementSet(HE={self[element]})
        else:
            condition = None
            if element_ndim == 1:
                condition = lambda x: x in Es or "any" in Es
            elif element_ndim > 1:
                condition = lambda x: all([x[i] in Es[i] or "any" in Es[i] for i in range(element_ndim)])
            return HElementSet(HE=set(el for el in self._HE.values() if condition(el.key)))


class Histogram2D(Histogram):
    """Data Histogram with 2D positioning"""
    pass


class Histogram3D(Histogram):
    """Data Histogram with 3D positioning"""
    pass


class OperationBase:
    """Base class for operations"""

    sign = None
    description = None

    def compute(self, arg1, arg2):
        raise NotImplementedError

    def __call__(self, arg1, arg2):
        return self.compute(arg1, arg2)


"""
Operations over Histogram
"""


class HistUnion(OperationBase):

    sign = "+"
    description = ""

    def compute(self, arg1, arg2):
        opn1, opn2 = (arg2, arg1) if len(arg1) > len(arg2) else (arg1, arg2)
        hist = Histogram(data=None)
        for key, value in opn2:
            hist[key] = HElement(key, value.value)
        for key, value in opn1:
            if key not in hist:
                hist[key] = HElement(key, 0)
            hist[key].value += value.value
        return hist


class HistIntersection:

    sign = "*"
    description = ""

    def compute(self, arg1, arg2):
        opn1, opn2 = (arg2, arg1) if len(arg1) > len(arg2) else (arg1, arg2)
        hist = Histogram(data=None)
        for key, value in opn1:
            if key in opn2:
                hist[key] = HElement(key, min(value.value, opn2[key].value))
        return hist


hist_operations = {
    HistUnion.sign: HistUnion(),
    HistIntersection.sign: HistIntersection()
}


"""
Operation over HElementSet
"""


class SetUnion(OperationBase):

    sign = "+"
    description = ""

    def compute(self, arg1, arg2):
        return HElementSet(arg1.union(arg2))


class SetIntersection(OperationBase):

    sign = "*"
    description = ""

    def compute(self, arg1, arg2):
        return HElementSet(arg1.intersection(arg2))


class SetSubtraction(OperationBase):

    sign = "/"
    description = ""

    def compute(self, arg1, arg2):
        return HElementSet(arg1.difference(arg2))


class SetXSubtraction(OperationBase):

    sign = "#/"
    description = ""

    def compute(self, arg1, arg2):
        return HElementSet() if arg2.sum() > 0 else arg1


class SetAndOr(SetUnion):
    sign = "|"


class SetOr(OperationBase):

    sign = "#|"
    description = ""

    def compute(self, arg1, arg2):
        return arg1 if arg1.sum() > arg2.sum() else arg2


class SetAnd(OperationBase):

    sign = "&"
    description = ""

    def compute(self, arg1, arg2):
        return arg2 if arg1.sum() > arg2.sum() else arg1


"""
Operation over HElementDict
"""


class DictUnion(OperationBase):

    def compute(self, arg1, arg2):
        opn1, opn2 = (arg2, arg1) if len(arg1) > len(arg2) else (arg1, arg2)
        for k, v in opn1:
            opn2.add(v)
        return opn2


operations = {
    SetUnion.sign: SetUnion(),
    SetIntersection.sign: SetIntersection(),
    SetAnd.sign: SetAnd(),
    SetOr.sign: SetOr(),
    SetAndOr.sign: SetAndOr(),
    SetSubtraction.sign: SetSubtraction(),
    SetXSubtraction.sign: SetXSubtraction()
}
