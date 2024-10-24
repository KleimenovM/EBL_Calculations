# General physical constants used in calculations

C: float = 299792458.0  # [m/s], speed of light, Source: https://en.wikipedia.org/wiki/Speed_of_light
H_EV: float = 4.135667696e-15  # [eV s], Planck's constant, Source: https://en.wikipedia.org/wiki/Planck_constant
H_J: float = 6.62607015e-34  # [J s], Planck's constant, Source: https://en.wikipedia.org/wiki/Planck_constant

SIGMA_TH: float = 6.65246 * 1e-29  # [m2], Source: https://en.wikipedia.org/wiki/Thomson_scattering
M_E: float = 510999.0  # [eV c-2], Source: https://en.wikipedia.org/wiki/Electron_mass
M_E2: float = M_E**2

J_EV: float = 1.602177e-19  # [eV/J], 1 eV in J, Source: https://en.wikipedia.org/wiki/Electronvolt
EV_J: float = J_EV**(-1)  # [J/eV], 1 J in eV

H0: float = 7.0e4  # [m/s Mpc-1], Hubble constant
OMEGA_DE: float = 0.7  # [DL], dark energy density
OMEGA_M: float = 0.3  # [DL], matter density

PC_M: float = 3.0857e16  # [m/Pc], Parsec in meters, Sourse: https://en.wikipedia.org/wiki/Parsec
MPC_M: float = PC_M * 1e6  # [m/MPc], 1 Mpc in m

K_B_J: float = 1.380649e-23  # [J K-1], Boltzmann constant, Source: https://en.wikipedia.org/wiki/Boltzmann_constant
K_B_EV: float = K_B_J * EV_J
