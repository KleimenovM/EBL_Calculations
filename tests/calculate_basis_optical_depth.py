from src.ebl_photon_density import EBLBasis
from src.functional_basis import BSplineBasis
from src.optical_depth import BasisOpticalDepth


def calculate_basis_optical_depth():
    fb = BSplineBasis(n=17)
    ebl_model = EBLBasis(basis=fb, cmb_on=True)
    bod = BasisOpticalDepth(ebl_model=ebl_model,
                            name="BSpline_17")
    bod.set()
    bod.save()
    return


if __name__ == '__main__':
    calculate_basis_optical_depth()
