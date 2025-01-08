
# This fork

Forked from [ankit7540/H2-PolarizabilityMatrixElements](https://github.com/ankit7540/H2-PolarizabilityMatrixElements). Published in [Polarizability tensor invariants of H<sub>2</sub>, HD, and D<sub>2</sub>](https://doi.org/10.1063/1.5011433).

This fork has the following changes relative to the original branch:
- Optional verbosity
- Installable as a package
- Caching of results

## Installation
Development version: `pip install git+https://github.com/Brightsite-Plasmalab/H2-PolarizabilityMatrixElements@develop`  
Latest stable version: `pip install git+https://github.com/Brightsite-Plasmalab/H2-PolarizabilityMatrixElements@master`

## Example usage
```python
from h2_rovib_me import *
import numpy as np
import matplotlib.pyplot as plt

# Calculate the intensity of the S-branch vibrational anti-Stokes Raman transition
# i.e. dv = -1, dJ = 2
dJ = 2
dv = -1

Ji = np.arange(0, 13) # (13,) = (1, 13)
Jf = Ji + dJ
vi = np.arange(1, 5)[:, np.newaxis] # (4, 1)
print(vi.shape)
vf = vi + dv

# Compute the isotropic and anisotropic contributions
iso = compute_batch("H2", vi, Ji, vf, Jf, 532, "nm", "iso") # (4, 13)
aniso = compute_batch("H2", vi, Ji, vf, Jf, 532, "nm", "aniso") # (4, 13)

for parameter, label in zip([iso, aniso], ["iso", "aniso"]):
    plt.figure()

    # Plot the intensity as a function of Ji for each vi
    for i, v in enumerate(vi):
        plt.plot(Ji, parameter[i], 's', label=f"$v_i$={v}")

    plt.xlabel("$J_i$")
    plt.ylabel(label)
    plt.legend()
    plt.show()
```

# H<sub>2</sub>-PolarizabilityMatrixElements

[Link to the article](https://doi.org/10.1063/1.5011433) {  Polarizability tensor invariants of H<sub>2</sub>, HD, and D<sub>2</sub>, https://doi.org/10.1063/1.5011433  }

Set of distance dependent data on polarizability together with FORTRAN and python programs for the interpolation (of polarizability over internuclear distance) and computation of the matrix elements over rovibrational states covering *J*=0--15 and *v*=0--4 within the ground electronic state. The programs evaluate the following integral:

![integral image][img0]

where, <img src="https://github.com/ankit7540/H2-PolarizabilityMatrixElements/blob/master/image/rmin.png" data-canonical-src="https://github.com/ankit7540/H2-PolarizabilityMatrixElements/blob/master/image/rmin.png" width="45" height="15" /> = 0.2 *a.u.* and  <img src="https://github.com/ankit7540/H2-PolarizabilityMatrixElements/blob/master/image/rmax.png" data-canonical-src="https://github.com/ankit7540/H2-PolarizabilityMatrixElements/blob/master/image/rmax.png" width="45" height="15" /> = 4.48 *a.u.*
This repository contains :
 - Inter-nuclear distance dependent polarizability,

Property | Definition
------------ | -------------
Polarizability perpendicular to the internuclear axis | <img src="https://github.com/ankit7540/H2-PolarizabilityMatrixElements/blob/master/image/alpha_perp.png" data-canonical-src="https://github.com/ankit7540/H2-PolarizabilityMatrixElements/blob/master/image/alpha_perp.png" width="30" height="15" />
Polarizability parallel to the internuclear axis | <img src="https://github.com/ankit7540/H2-PolarizabilityMatrixElements/blob/master/image/alpha_parallel.png" data-canonical-src="https://github.com/ankit7540/H2-PolarizabilityMatrixElements/blob/master/image/alpha_parallel.png" width="23" height="20" />
Mean polarizability | <img src="https://github.com/ankit7540/H2-PolarizabilityMatrixElements/blob/master/image/alpha_mp.png" data-canonical-src="https://github.com/ankit7540/H2-PolarizabilityMatrixElements/blob/master/image/alpha_mp.png" width="195" height="28" />
Polarizability anisotropy | <img src="https://github.com/ankit7540/H2-PolarizabilityMatrixElements/blob/master/image/gamma.png" data-canonical-src="https://github.com/ankit7540/H2-PolarizabilityMatrixElements/blob/master/image/gamma.png" width="155" height="28" />

The above properties are available as `Omega` in the above integral for H<sub>2</sub>, HD and D<sub>2</sub>.
 - Rovibrational wavefunctions for H<sub>2</sub>, HD and D<sub>2</sub> for v=0 - 4 and J=0 - 15.
 - A FORTRAN program and a python module which can be used to compute the static and wavelength dependent matrix elements. Wavelength range available is 182.25 to 1320.6 nm.

---

This repository provides software for computing ro-vibrational matrix elements of polarizability invariants which can be used to obtain :

 - Wavelength-dependent Rayleigh intensities and corresponding depolarization ratios
 - Wavelength-dependent Raman intensities and corresponding depolarization ratios
 - Wavelength-dependent refractive index
 - Verdet constant
 - Kerr constant


**Available programs**
---
The programs for computation of matrix element (which includes cubic spline interpolation and numerical integration) are written in FORTRAN and Python. These are independent programs which do the same job.

In the case of FORTRAN, two different programs exist, *(i)* `rovibME_dynamic.f` for wavelength dependent matrix elements, and *(ii)* `rovibME_static.f` for static ones.

In the case of Python, one program `rovibME.py` deals with both static and dynamic matrix elements. [See example](https://github.com/ankit7540/H2-PolarizabilityMatrixElements/blob/master/python-module/example/H2_polarizability_MEs_example.ipynb) for more details.

**Usage**
---
Clone this repository or download as a zip file. According to the program of choice, refer to the `README.md` in the FORTRAN-program folder or in the Python-module folder. (Both versions do the same computation and give same results.)


**Comments on numerical accuracy**
---
The definite integral calculation is usually accurate to ~1e-6 or better. However, the net numerical uncertainty in the computed matrix element is  +/- 1e-4 which includes the uncertainties introduced by the accuracy of the wavefunctions, polarizability, spline interpolation procedures and physical constants.

**Comments on the sign of the matrix element**
---
Some matrix elements computed may have negative sign which arises due to the phase of the wavefunction. In most applications, the square of the matrix elements are needed and thus the sign maybe of no real consequence.


**Credits**
---
Cubic spline interpolation procedure used in FORTRAN and python codes has been adapted from Numerical Recipes in FORTRAN, William H. Press, Saul A. Teukolsky, William T. Vetterling, Brian P. Flannery, Michael Metcalf, Cambridge University Press; 2<sup>nd</sup> edition.

For evaluation of the definite integral the Adaptive Gausssian Quadrature implemented in SciPy has been used.

**References on the evaluation of the definite integral and implementation:**
- T. N. L. Patterson, <i>Math. Comput.</i> 22, 847 (1968)
- T. N. L. Patterson, <i>Math. Comput.</i> 23, 892 (1969)
- R. Piessens, E. de Doncker-Kapenga, C. Uberhuber, and D. Kahaner, Quadpack - A Sub-routine Package for Automatic Integration (Springer-Verlag Berlin Heidelberg, 1983)


FORTRAN code by Prof. Henryk A. Witek (NYCU, Taiwan).

Python code by Ankit Raj (NYCU, Taiwan).

---

**This work has been published in the following article:**

**Polarizability tensor invariants of H<sub>2</sub>, HD, and D<sub>2</sub> <br>**
Raj, A., Hamaguchi, H., and Witek, H. A.<br>
<em><i>Journal of Chemical Physics </i></em><strong>148</strong>, 104308 (2018) <br>
<a href="https://aip.scitation.org/doi/abs/10.1063/1.5011433">10.1063/1.5011433	</a>

---

Application oriented research based on this repository:

 - Vibration–rotation interactions in H2, HD and D2 : centrifugal distortion factors and the derivatives of polarisability invariants ([10.1080/00268976.2019.1632950 ](https://www.tandfonline.com/doi/full/10.1080/00268976.2019.1632950))
 - Toward standardization of Raman spectroscopy: Accurate wavenumber and intensity calibration using rotational Raman spectra of H<sub>2</sub>, HD, D<sub>2</sub>, and vibration–rotation spectrum of O<sub>2</sub> ([10.1002/jrs.5955](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/full/10.1002/jrs.5955))

 - Determination of accurate absolute Raman cross-section of benzene and cyclohexane in the gas phase (Asian J.Phys., 30 (2021) 321-335.)

 - Accurate intensity calibration of multichannel spectrometers using Raman intensity ratios ([10.1002/jrs.6221](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/jrs.6221))



[img0]: https://github.com/ankit7540/H2-PolarizabilityMatrixElements/blob/master/image/01-05-2018_82.png "Logo Title Text 2"
[img1]: https://github.com/ankit7540/H2-PolarizabilityMatrixElements/blob/master/image/alpha_perp.png "Logo alpha_{perp}"
[img2]: https://github.com/ankit7540/H2-PolarizabilityMatrixElements/blob/master/image/alpha_parallel.png "Logo alpha_{paralell}"
[img3]: https://github.com/ankit7540/H2-PolarizabilityMatrixElements/blob/master/image/alpha_mp.png "Logo alpha_{mp}"
[img4]: https://github.com/ankit7540/H2-PolarizabilityMatrixElements/blob/master/image/gamma.png "Logo alpha_{aniso}"
