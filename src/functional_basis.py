import numpy as np


class FunctionalBasis:
    """
    A parent class which describes functional basis sets for fitting the EBL density
    """
    def __init__(self, n: int = 5, m: int = 100):
        """
        Initialize the functional basis set
        :param n: number of basis functions
        :param m: number of points in standard lg_wvl split
        """
        self.n = n  # number of basis functions
        self.m = m  # standard wvl_split size

        self.low_lg_wvl: float = 0.0  # [DL], lg(wvl/mkm)
        self.high_lg_wvl: float = 3.0  # [DL], lg(wvl/mkm)

        self.fb = []

    def get_lg_wvl_range(self, m: int):
        """
        Get a range of lg_wvl
        :param m: number of points in the desired lg_wvl split
        :return: <np.ndarray> [DL] - a homogenous range of lg_wvl
        """
        return np.linspace(self.low_lg_wvl, self.high_lg_wvl, m)

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
            lg_wvl = self.get_lg_wvl_range(m)
        elif lg_wvl is not None:
            m = lg_wvl.size
        elif lg_wvl.size != m:
            raise ValueError("Choose either lg_wvl or m!")

        result = np.zeros((self.n, m))

        for i in range(self.n):
            result[i, :] = self.fb[i](lg_wvl)
        return lg_wvl, result


class BasisFunction:
    def __init__(self, i: int, f):
        self.i = i
        self.f = f

    def __call__(self, lg_wvl):
        return self.f(lg_wvl, self.i)


class ParabolicBasis(FunctionalBasis):
    def __init__(self, n: int, m: int):
        super().__init__(n, m)
        edges = np.linspace(self.low_lg_wvl, self.high_lg_wvl, self.n + 1)  # the dots in .|.|.|.|.
        self.peaks = (edges[:-1] + edges[1:]) / 2.0  # the columns in .|.|.|.|.
        self.delta = self.peaks[1] - self.peaks[0]  # distance between the peaks (mutual for all the peaks)
        self.sigma2 = self.delta ** 2 / (8 * np.log(2))  # hwfh
        self.set_function_list()

    def an_exponentiated_parabola(self, lg_wvl, i):
        return np.exp(-(lg_wvl - self.peaks[i])**2 / (2 * self.sigma2))

    def set_function_list(self):
        for i in range(self.n):
            bf = BasisFunction(i=i, f=self.an_exponentiated_parabola)
            print(self.peaks[i])
            self.fb.append(bf)
        pass
