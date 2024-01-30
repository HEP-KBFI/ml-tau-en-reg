import operator
import numpy as np

OPERATORS = {">=": operator.ge, "<=": operator.le, "==": operator.eq, ">": operator.gt, "<": operator.lt}


class GeneralCut:
    """Cuts based on string"""

    def __init__(self, cut_string):
        """Initializes the string based cut class

        Args:
            cut_string : str
                String containing all the cuts in a default convention

        Returns:
            None
        """
        self.cut_string = cut_string
        self._all_cuts = []
        self.separate_cuts()

    def separate_cuts(self):
        """Separates all the cuts in the general cut string"""
        self.cut_string = self.cut_string.replace(" ", "")
        raw_cuts = self.cut_string.split("&&")
        for cut_ in raw_cuts:
            self._all_cuts.append(self.interpret_single_cut_string(cut_))

    def interpret_single_cut_string(self, cut_string):
        """Interpretes the single-cut string"""
        cut = ""
        for operator_ in OPERATORS.keys():
            if operator_ in cut_string:
                cut = cut_string.split(operator_)
                cut.insert(1, operator_)
                return cut
        if cut == "":
            raise ValueError("No cut selected")

    @property
    def all_cuts(self):
        return self._all_cuts


class Histogram:
    """Initializes the histogram"""

    def __init__(
        self,
        data: np.array,
        bin_edges: np.array,
        histogram_data_type: str,
        binned: bool = False,
        uncertainties: np.array = True,
    ) -> None:
        self.data = np.array(data)
        self.histogram_data_type = histogram_data_type
        self.bin_edges = bin_edges
        self.bin_centers, self.bin_halfwidths = self.calculate_bin_centers(bin_edges)
        if not binned:
            self.binned_data = np.histogram(data, bins=bin_edges)[0]
        else:
            self.binned_data = data
        if (type(uncertainties) == np.array) or (type(uncertainties) == np.ndarray):
            if len(uncertainties) == len(self.binned_data):
                self.uncertainties = uncertainties
            else:
                raise AssertionError(
                    f"Incorrect number of entries for uncertainties {len(uncertainties)} != {len(self.binned_data)}"
                )
        elif (type(uncertainties) == bool) and uncertainties:
            self.uncertainties = 1 / np.sqrt(self.binned_data)
        else:
            raise AssertionError("Unknown input for uncertainties")

    def calculate_bin_centers(self, edges: list) -> np.array:
        bin_widths = np.array([edges[i + 1] - edges[i] for i in range(len(edges) - 1)])
        bin_centers = []
        for i in range(len(edges) - 1):
            bin_centers.append(edges[i] + (bin_widths[i] / 2))
        return np.array(bin_centers), bin_widths / 2

    def __add__(self, other):
        if (other.bin_edges).all() != (self.bin_edges).all():
            raise ArithmeticError("The bins of two histograms do not match, cannot sum them.")
        result = self.binned_data + other.binned_data
        uncertainties = self.uncertainties + other.uncertainties
        return Histogram(result, self.bin_edges, "Sum", binned=True, uncertainties=uncertainties)

    def __str__(self):
        return f"{self.histogram_data_type} histogram"

    def __truediv__(self, other):
        if (other.bin_edges).all() != (self.bin_edges).all():
            raise ArithmeticError("The bins of two histograms do not match, cannot divide them.")
        result = np.nan_to_num(self.binned_data / other.binned_data, copy=True, nan=0.0, posinf=None, neginf=None)
        rel_uncertainties = np.sqrt(np.abs(result * (1 - result) / other.binned_data))  # use binomial errors
        # rel_uncertainties = (other.uncertainties / other.binned_data) + (self.uncertainties / self.binned_data)  # Poisson
        return Histogram(result, self.bin_edges, "Efficiency", binned=True, uncertainties=rel_uncertainties)

    def __mul__(self, other):
        if (other.bin_edges).all() != (self.bin_edges).all():
            raise ArithmeticError("The bins of two histograms do not match, cannot multiply them.")
        result = self.binned_data * other.binned_data
        rel_uncertainties = (other.uncertainties / other.binned_data) + (self.uncertainties / self.binned_data)
        return Histogram(result, self.bin_edges, "Multiplicity", binned=True, uncertainties=rel_uncertainties)
