import numpy as np
from numba import njit


class FunctionalBasis:
    """
    A parent class which describes functional basis sets for fitting the EBL density
    """
    def __init__(self, n: int = 5, m: int = 10000):
        """
        Initialize the functional basis set
        :param n: number of basis functions
        :param m: number of points in standard lg_wvl split
        """
        self.n = n  # number of basis functions
        self.m = m  # standard wvl_split size

        self.low_lg_wvl: float = -1.0  # [DL], lg(wvl/mkm)
        self.high_lg_wvl: float = 3.0  # [DL], lg(wvl/mkm)

        self.fb = []

    def get_lg_wvl_range(self, m: int = None):
        """
        Get a range of lg_wvl
        :param m: number of points in the desired lg_wvl split
        :return: <np.ndarray> [DL] - a homogenous range of lg_wvl
        """
        if m is None:
            m = self.m
        return np.linspace(self.low_lg_wvl, self.high_lg_wvl, m)

    def get_wvl_range(self, m: int = None):
        """
        Get a range of wvl
        :param m: number of points in the desired wvl split
        :return: <np.ndarray> [DL] - a logarithmically scaled range of wvl
        """
        return 10**self.get_lg_wvl_range(m)

    def set_function_list(self):
        """
        Set a list of basis functions (unique for each basis type)
        and fill the self_fb values
        """
        pass

    def get_function_list(self):
        """
        Get the basis function list
        :return:
        """
        if len(self.fb) != self.n:
            raise ValueError('Functional basis is not set!')
        return self.fb

    def get_distribution_list(self, lg_wvl: np.linspace = None, m: int = None):
        """
        Get the basis distributions
        :param lg_wvl: <optional> [DL], lg(wvl/mkm), list of wavelength
        :param m: <optional> size of the standard lg_wvl split
        :return: wavelength splitting and corresponding intensity distribution [W m-2 sr-1]
        """
        if len(self.fb) != self.n:
            raise ValueError('Functional basis is not set!')
        if lg_wvl is None and m is None:
            lg_wvl = self.get_lg_wvl_range(self.m)
        elif m is not None:
            self.m = m
            lg_wvl = self.get_lg_wvl_range(m)
        elif lg_wvl is not None:
            self.m = lg_wvl.size
        elif lg_wvl.size != m:
            raise ValueError("Choose either lg_wvl or m!")

        result = np.repeat([np.zeros_like(lg_wvl)], self.n, axis=0)

        for i in range(self.n):
            result[i] = self.fb[i](lg_wvl)
        return lg_wvl, result


class BasisFunction:
    def __init__(self, i: int, f):
        self.i = i
        self.f = f

    def __call__(self, lg_wvl):
        return self.f(lg_wvl, self.i)


class ExpParabolicBasis(FunctionalBasis):
    def __init__(self, n: int = 5, m: int = 10000):
        super().__init__(n, m)
        edges = np.linspace(self.low_lg_wvl, self.high_lg_wvl, self.n + 1)  # the dots in .|.|.|.|.
        self.peaks = (edges[:-1] + edges[1:]) / 2.0  # the columns in .|.|.|.|.
        self.h = edges[1] - edges[0]
        self.delta = self.peaks[1] - self.peaks[0]  # distance between the peaks (mutual for all the peaks)
        self.sigma2 = self.delta ** 2 / (8 * np.log(2))  # hwfh^2
        self.set_function_list()

    def an_exponentiated_parabola(self, lg_wvl, i):
        return np.exp(-(lg_wvl - self.peaks[i])**2 / (2 * self.sigma2))

    def set_function_list(self):
        for i in range(self.n):
            bf = BasisFunction(i=i, f=self.an_exponentiated_parabola)
            self.fb.append(bf)
        pass


class BSplineBasis(FunctionalBasis):
    def __init__(self, n: int = 5, m: int = 10000):
        super().__init__(n, m)
        self.knots = np.linspace(self.low_lg_wvl, self.high_lg_wvl, self.n + 2)  # the knots of the grid
        self.h = self.knots[1] - self.knots[0]  # distance between the knots
        self.set_function_list()

    def box(self, lg_wvl, i):
        eps = 10 * np.finfo(float).eps
        return np.heaviside(lg_wvl + eps - self.knots[i], 0.5) * np.heaviside(self.knots[i+1] - (lg_wvl + eps), 0.5)

    def b_spline_basis_function(self, lg_wvl, i):
        i += 1
        u = (lg_wvl - self.knots[i]) / self.h

        boxes = np.array([self.box(lg_wvl + 2 * self.h, i),
                          self.box(lg_wvl + 1 * self.h, i),
                          self.box(lg_wvl + 0 * self.h, i),
                          self.box(lg_wvl - 1 * self.h, i)])

        u0, u1, u2, u3 = u**0, u, u**2, u**3

        f1 = 8 * u0 + 12 * u1 + 6 * u2 + u3
        f2 = 4 * u0 - 6 * u2 - 3 * u3
        f3 = 4 * u0 - 6 * u2 + 3 * u3
        f4 = 8 * u0 - 12 * u1 + 6 * u2 - u3

        vector = np.array([f1, f2, f3, f4])

        return 1 / 4 * np.sum(vector * boxes, axis=0)

    def set_function_list(self):
        for i in range(self.n):
            bf = BasisFunction(i=i, f=self.b_spline_basis_function)
            self.fb.append(bf)
        return
