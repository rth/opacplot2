import numpy as np
import math

# Convert Joules to Ergs:
JOULE_TO_ERG = 1.0e+07

# Ergs to Joulse:
ERG_TO_JOULE = 1.0/JOULE_TO_ERG

# Convert GPa to Ergs/cc:
GPA_TO_ERGCC = 1.0e+10

# Convert MJ/kg to Ergs/g:
MJKG_TO_ERGG = 1.0e+10

# Convert MJ/kg to Ergs/cc:
MJKG_TO_ERGCC = 1.0e+10

# Convert  Ergs/cc to GPa:
ERGCC_TO_GPA =1./GPA_TO_ERGCC

# Convert  Ergs/g to MJ/kg:
ERGG_TO_MJKG =1./MJKG_TO_ERGG

# Convert Kelvin to eV:
KELVIN_TO_EV = 1.0/11604.55

# Convert eV to Kelvin:
EV_TO_KELVIN = 11604.55

# Avogadros number
NA = 6.0221415e+23

# Boltzmann constant [ergs/eV]
KB = 1.60217653e-12

# Electron mass [g]
ME = 9.10938188e-28

# Planck's Constants [erg*s]:
PLANCK = 6.62606957e-27
HBAR = PLANCK/(2*math.pi)

# Radiation constant [ergs/cm^3/eV^4]
RADCONST = 137.201730837

BC_BOUND       = 0
BC_EXTRAP      = 1
BC_EXTRAP_ZERO = 2

INTERP_FUNC = 0
INTERP_DFDD = 1
INTERP_DFDT = 2

MAX_ELE = 100

ELE_HE = 4
ELE_AL = 13

# Atomic Weights:
ATOMIC_WEIGHTS = -1 * np.ones(MAX_ELE)

ATOMIC_WEIGHTS[ELE_HE] = 4.002602
ATOMIC_WEIGHTS[ELE_AL] = 26.9815386


# Atomic Symbols:
ATOMIC_SYMBOLS = ["H" , "He", "Li", "Be", "B", "C",
                  "N" , "O" ]
