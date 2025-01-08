# Purpose : Load the matrix containing polarizability and wavefunctions, interpolate the
# polarizability if needed, and compute the respective matrix elements.

# Load necessary modules
import sys
import numpy as np
from scipy import interpolate
from scipy import integrate
import pickle as pkl
from pathlib import Path

dir_root = Path(__file__).parent
dir_data = dir_root / "data"
dir_wave = dir_root / "wavefunctions"

# -------------------------------------------------------------------

# ********************************************************************
# python function to compute the first derivative at the first point
# and the last point of an array. For this computation, the first 4
# points are used for the derivative for the first data point. Similarly
# last 4 (x,y) points are used for the derivative at the last point.


def fd_ends(x, y):
    """Parameter:
    x       =       xaxis of the array
    y       =       y data of the array
            (x and y arrays must have atleast 4 elements)
    Returns = first derivative at the first point and the last point
    """
    if len(x) < 4 or len(y) < 4:
        print("Error : x and y arrays must have 4 elements")

    subx = np.zeros(4)
    suby = np.zeros(4)

    for i in range(0, 4):
        subx[i] = x[i]
        suby[i] = y[i]

    fd1 = (
        (
            subx[1] * subx[3]
            + subx[2] * subx[3]
            + subx[1] * subx[2]
            - 2 * subx[0] * subx[1]
            - 2 * subx[0] * subx[3]
            - 2 * subx[0] * subx[2]
            + 3 * subx[0] ** 2
        )
        / (-subx[3] + subx[0])
        / (-subx[1] + subx[0])
        / (-subx[2] + subx[0])
        * suby[0]
        - (-subx[2] + subx[0])
        * (-subx[3] + subx[0])
        / (subx[1] - subx[3])
        / (-subx[1] + subx[0])
        / (subx[1] - subx[2])
        * suby[1]
        + (-subx[1] + subx[0])
        * (-subx[3] + subx[0])
        / (subx[2] - subx[3])
        / (subx[1] - subx[2])
        / (-subx[2] + subx[0])
        * suby[2]
        - (-subx[1] + subx[0])
        * (-subx[2] + subx[0])
        / (subx[2] - subx[3])
        / (subx[1] - subx[3])
        / (-subx[3] + subx[0])
        * suby[3]
    )

    for i in range(0, 4):
        subx[i] = x[int(i - 4)]
        suby[i] = y[int(i - 4)]
    #        print (i, int(i-4))

    fdn = (
        (subx[1] - subx[3])
        * (subx[2] - subx[3])
        / (-subx[3] + subx[0])
        / (-subx[1] + subx[0])
        / (-subx[2] + subx[0])
        * suby[0]
        - (-subx[3] + subx[0])
        * (subx[2] - subx[3])
        / (subx[1] - subx[3])
        / (-subx[1] + subx[0])
        / (subx[1] - subx[2])
        * suby[1]
        + (-subx[3] + subx[0])
        * (subx[1] - subx[3])
        / (subx[2] - subx[3])
        / (subx[1] - subx[2])
        / (-subx[2] + subx[0])
        * suby[2]
        - (
            -2 * subx[0] * subx[3]
            - 2 * subx[1] * subx[3]
            - 2 * subx[2] * subx[3]
            + subx[0] * subx[1]
            + subx[0] * subx[2]
            + subx[1] * subx[2]
            + 3 * subx[3] ** 2
        )
        / (subx[2] - subx[3])
        / (subx[1] - subx[3])
        / (-subx[3] + subx[0])
        * suby[3]
    )

    return (fd1, fdn)


# ********************************************************************


# Spline function taken from Numerical Recipes in FORTRAN, page 109 and 110
# Numerical recipes in FORTRAN, Second Edition, Press, Teukolsky, Vetterling, Flannery
# Cambridge University Press, 1992
def spline(x, y, yp1, ypn):
    """Parameters:
    x       =       1D vector of x-values in increasing order.
    y       =       1D vector of y-values
    n       =       number of elements in xVector (and yVector)
    yp1     =       first derivative of the interpolating function at the first segment
    ypn     =       first derivative of the interpolating function at the last segment

    """
    nx = len(x)
    ny = len(y)

    if nx == ny:
        n = nx
    else:
        print("Error : x and y data have different lengths in spline.")
        quit()

    u = np.zeros(n)
    y2 = np.zeros(n)  # this is the output
    p = 0.0
    sig = 0.0

    if yp1 > 1e30:  # lower boundar condition 'natural'
        y2[0] = 0.0
        u[0] = 0.0
    else:  # specified first derivative
        y2[0] = -0.5
        u[0] = (3 / (x[1] - x[0])) * (((y[1] - y[0]) / (x[1] - x[0])) - yp1)

    for i in range(
        1, n - 1
    ):  #       Decomposition loop of tridiagonal algorithm. y2 and u are temporary.
        sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1])
        p = (sig * y2[i - 1]) + 2.0
        y2[i] = (sig - 1.0) / p
        u[i] = (
            6.0
            * (
                (y[i + 1] - y[i]) / (x[i + 1] - x[i])
                - (y[i] - y[i - 1]) / (x[i] - x[i - 1])
            )
            / (x[i + 1] - x[i - 1])
            - sig * u[i - 1]
        ) / p
        # print("first loop:",i)

    if ypn > 1e30:  # upper boundary condition 'natural'
        qn = 0.0
        un = 0.0
    else:  # specified first derivative
        qn = 0.5
        un = (3.0 / (x[n - 1] - x[n - 2])) * (
            ypn - ((y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]))
        )

    y2[n - 1] = (un - qn * u[n - 2]) / (qn * y2[n - 2] + 1.0)

    for k in range(n - 2, -1, -1):  # from second last point to the second point
        y2[k] = (
            y2[k] * y2[k + 1] + u[k]
        )  # backsubstitution loop of tridiagonal algorithm
        # print("loop 2 :",k)

    return y2


# ************************************************************************
# Spline interpolation function taken from Numerical Recipes in FORTRAN, page 109 and 110
# Numerical recipes in FORTRAN, Second Edition, Press, Teukolsky, Vetterling, Flannery
# Cambridge University Press, 1992


def splint(xa, ya, y2a, x):
    """Parameters :
    xa      =       original x-axis 1D vector
    ya      =       original y-axis 1D vector
    y2a     =       output of the spline function
    x       =       new x axis, scalar
    """

    nxa = len(xa)
    nya = len(ya)
    ny2a = len(y2a)

    if nxa != nya or nxa != ny2a or nya != ny2a:
        print("Error : xa or ya or y2a have incorrect dimension(s).")
        quit()

    n = nxa

    klo = int(0)
    khi = int(n - 1)
    k = int(0)
    h = 0.0
    element = 0.0

    while (khi - klo) > 1:
        k = int((khi + klo) / 2)
        element = xa[k]
        #        print(element,xa[k],k,x)
        if element > x:
            khi = k
        else:
            klo = k

    h = xa[khi] - xa[klo]

    if h == 0:
        print("Error : Bad xa input in splint")
        quit()

    a = (xa[khi] - x) / h
    b = (x - xa[klo]) / h
    y = (
        a * ya[klo]
        + b * ya[khi]
        + (((a**3) - a) * y2a[klo] + ((b**3) - b) * y2a[khi]) * (h**2) / 6.0
    )

    return y  # returns the interpolated value


# ************************************************************************
# ************************************************************************


def help():
    omega = np.loadtxt(dir_data / "freq.txt")
    print("Polarizability data dimension checked.")
    print("\n")
    omega_nm = 1e7 / (omega * 219474.6313702000)
    omega_A = omega_nm * 10
    print("Give  rovibME.compute  command with parameters:")
    print(
        "\trovibME.compute(molecule, bra_v, bra_J, ket_v, ket_J, lambda, unit of lambda, operator)"
    )
    print('\t for example:  rovibME.compute("H2",0,2,0,4,488,"n","mp")  ')
    print('\t\t       rovibME.compute("D2",1,0,1,0,"static","n","all")  ')
    print("\t\t")

    print('\t\tmolecule = for H2 enter "H2", for D2 enter "D2", for HD enter "HD" ')
    print("\t\tbra_v    = vibrational state, v=[0,4]")
    print("\t\tbra_J    = rotataional state, J=[0,15]")
    print("\t\tket_v    = vibrational state, v=[0,4]")
    print("\t\tket_J    = rotataional state, J=[0,15]")
    print(
        '\t\tlambda   = wavelength in Hartree, nm or Angstrom, for static specify "s" or "static" here'
    )
    print('\t\tunit of lambda =  for  Hartree           use "H" or "h"  ')
    print('\t\t\t          for  nanometers        use "n" or "nm" ')
    print('\t\t\t          for  Angstrom          use "a" or "A"  ')
    print(
        "\t\t\t          if static property is asked then this parameter can be any of the three "
    )
    print(
        "\t\tAvailable wavelength range: {0} - {1} Hartree;\n   \t\t\t\t\t    {2} - {3} nm; \n   \t\t\t\t\t    {4} - {5} Angstrom".format(
            round(omega[0], 4),
            round(omega[-1], 4),
            round(omega_nm[0], 4),
            round(omega_nm[-1], 4),
            round(omega_A[0], 4),
            round(omega_A[-1], 4),
        )
    )
    print('\t\toperator	= alpha(perpendicular) = alpha_xx given by "xx" or "x" ')
    print('\t\t\t          alpha(parallel) = alpha_xx given by "zz" or "z" ')
    print(
        '\t\t\t          isotropy or mean polarizability given by "iso" or "mp" or "mean" '
    )
    print(
        '\t\t\t          anisotropy or polarizability difference or gamma given by "aniso" or "g"  or "diff" '
    )
    print('\t\t\t          for all the above use "all"   or  "All"or "ALL" ')

    print("...ready.")


# ********************************************************************
# the actual function for computation of the rovibrational matrix element.
# vl, Jl, vr , Jr are numbers
# mol, wavelength unit and operator are string, hence need quotes.


def compute(mol, vl, Jl, vr, Jr, wavelength, wavelength_unit, operator, verbose=True):
    r"""#  parameters:
    # mol  =    molecule (for H2 enter "H2", for D2 enter "D2", for HD enter "HD")
    # vl   =    vibrational state for the bra, vl = [0,4]
    # Jl   =    rotational state for the bra,  Jl = [0,15]
    # vr   =    vibrational state for the ket, vr = [0,4]
    # Jr   =    rotational state for the ket,  Jr = [0,15]
    # wavelength =  wavelength ( can be Hartree, nanometers or Angstrom)
    # wavelength_unit = specify unit using the specifier
                                ( for  Hartree           use "H" or "h"  )
                                ( for  nanometers        use "n" or "nm"  )
                                ( for  Angstrom          use "a" or "A"  )

    # operator   = property namely alpha_xx, alpha_zz, mean polarizability
                                   (isotropy)[\bar{alpha}], anisotropy[\gamma]
                                   Specify operator using the specifier.
                                 ( for  alpha_xx = alpha(?)         use "x"     or  "xx"  )
                                 ( for  alpha_zz          use "z"     or  "zz"  )
                                 ( for  isotropy          use "iso"   or  "mp" or "mean" )
                                 ( for  anisotropy        use "aniso" or  "g"  or "diff" )
                                 ( for  all the above     use "all"   or  "All"or "ALL"  )

    This function runs on both Python 2.7x and 3.x
    """

    # set a dictionary for output array
    d = {"output": [0]}

    # ----------------------------------------------------------------
    # interpolation function defined here and is used later.
    def interpolate2D_common(input2D, originalx, finalx):
        inputSize = input2D.shape
        tempx = np.zeros((len(originalx), 1))
        tempx[:, 0] = originalx
        col = np.zeros((len(originalx), 1))
        outputArray = np.zeros((1, inputSize[1]))
        for i in range(0, inputSize[1]):
            col[:, 0] = input2D[:, i]
            der = (0.0, 0.0)  # derivatives at first and last ends
            der = fd_ends(tempx, col)
            # print(i,der[0],der[1])
            secarray = spline(tempx, col, der[0], der[1])
            interp = splint(tempx, col, secarray, finalx)
            outputArray[:, i] = interp
        d["output"] = outputArray.T

    # ----------------------------------------------------------------

    # Load the polarizability data ( alpha_xx and alpha_zz)
    alpha_xx = np.loadtxt(dir_data / "matrix_xxf.txt")
    alpha_zz = np.loadtxt(dir_data / "matrix_zzf.txt")
    omega = np.loadtxt(dir_data / "freq.txt")
    dist = np.loadtxt(dir_data / "distance.txt")
    distance = np.asarray(dist)
    omega_nm = 1e7 / (omega * 219474.6313702000)  # convert the original freq to nm
    static_xx = np.loadtxt(dir_data / "static_xx.txt")
    static_zz = np.loadtxt(dir_data / "static_zz.txt")

    if not (
        alpha_xx.shape == alpha_zz.shape
        or len(omega) == alpha_xx.shape[0]
        or len(static_xx) == len(distance)
        or len(static_zz) == len(distance)
    ):
        raise Exception(
            f"Dimension check on polarizability data matrices or wavelength file failed. Please check that the files in {dir_data} are correct."
        )

    # compute the isotropy(mean polarizability) and anisotropy (gamma)
    isotropy = np.absolute(2 * (np.array(alpha_xx)) + np.array(alpha_zz)) / 3
    anisotropy = np.absolute(np.array(alpha_zz) - np.array(alpha_xx))

    isotropy_static = (2 * static_xx + static_zz) / 3
    anisotropy_static = static_zz - static_xx

    # step 1: load the required wavefunctions ------------------------
    Wfn1 = dir_wave / "{0}v{1}J{2}_norm.txt".format(mol, vl, Jl)
    Wfn2 = dir_wave / "{0}v{1}J{2}_norm.txt".format(mol, vr, Jr)
    r_wave = dir_wave / "r_wave.txt"
    # print(Wfn1,Wfn2)
    if vl < 0 or vr < 0 or vl > 4 or vr > 4:
        print("Error : v value out of range. vl and vr = [0,4]. Exiting ")
        quit()

    if Jl < 0 or Jr < 0 or Jl > 15 or Jr > 15:
        print("Error : J value out of range. Jl and Jr =[0,15]. Exiting ")
        quit()

    if not (mol == "H2" or mol == "HD" or mol == "D2"):
        print(
            "Error : Incorrect molecule chosen. For H2 enter H2, for D2 enter D2, for HD enter HD. Use quotes. Exiting  "
        )
        quit()

    # Proceed to load wavefunctions.
    psi1 = np.loadtxt(Wfn1)
    psi2 = np.loadtxt(Wfn2)
    rwave = np.loadtxt(r_wave)
    # print(len(psi1),len(psi2),len(rwave))
    # ----------------------------------------------------------------
    # STATIC
    if wavelength == "static" or wavelength == "s":
        print("\tStatic")
        n = 0
        if operator == "x" or operator == "xx":
            param = static_xx
            name = ["alpha_xx"]
            n = 1
        elif operator == "z" or operator == "zz":
            param = static_zz
            name = ["alpha_zz"]
            n = 1
        elif operator == "mean" or operator == "mp" or operator == "iso":
            param = isotropy_static
            name = ["isotropy"]
            n = 1
        elif operator == "diff" or operator == "g" or operator == "aniso":
            param = anisotropy_static
            name = ["anisotropy"]
            n = 1
        elif operator == "all" or operator == "All" or operator == "ALL":
            list = [static_xx, static_zz, isotropy_static, anisotropy_static]
            name = ["alpha_xx", "alpha_zz", "isotropy", "anisotropy"]
            n = 4
        else:
            print("Error : Operator not correctly specified. Exiting ")
            quit()

    # DYNAMIC
    elif isinstance(wavelength, (int, float)):
        wv = float(wavelength)  # entered wavelength is a number

        if wavelength_unit == "h" or wavelength_unit == "H":
            omegaFinal = 1e7 / (wv * 219474.6313702000)
        elif wavelength_unit == "n" or wavelength_unit == "nm":
            omegaFinal = wv
        elif wavelength_unit == "a" or wavelength_unit == "A":
            omegaFinal = wv / 10
        else:
            print("Message : Default unit of nm will be used.")
            omegaFinal = wv

        if omegaFinal < omega_nm[0] or omegaFinal > omega_nm[-1]:
            sys.exit("Error : Requested wavelength is out of range. Exiting ")

        if verbose:
            print(
                "Selected wavelength in nanometer : {0}, Hartree : {1}".format(
                    round(omegaFinal, 6),
                    round((1e7 / (omegaFinal * 219474.63137020)), 6),
                )
            )

        n = 0
        if operator == "x" or operator == "xx":
            param = alpha_xx
            name = ["alpha_xx"]
            n = 1
        elif operator == "z" or operator == "zz":
            param = alpha_zz
            name = ["alpha_zz"]
            n = 1
        elif operator == "mean" or operator == "mp" or operator == "iso":
            param = isotropy
            name = ["isotropy"]
            n = 1
        elif operator == "diff" or operator == "g" or operator == "aniso":
            param = anisotropy
            name = ["anisotropy"]
            n = 1
        elif operator == "all" or operator == "All" or operator == "ALL":
            list = [alpha_xx, alpha_zz, isotropy, anisotropy]
            name = ["alpha_xx", "alpha_zz", "isotropy", "anisotropy"]
            n = 4
        else:
            print("Error : Operator not correctly specified. Exiting")
            quit()

    else:
        print(
            'Error : Incorrect specification of wavelength. Use number for dynamic property and "s" or "static" for static. Exiting'
        )
        quit()

    # -------------------------------------------------------------------------
    for i in range(n):  # evaluation of  interpolation and integral

        # step 1: prepare parameter vector(s)
        parameter = np.zeros((len(distance), 1))
        # Static
        if wavelength == "static" or wavelength == "s":
            if not (n == 1):
                parameter = list[i]
            else:
                parameter = param
        else:
            # interpolate to the asked wavelength ----------------
            if not (n == 1):
                param = list[i]
                # print(param.shape, omegaFinal)
                interpolate2D_common(param, omega_nm, omegaFinal)
            else:
                interpolate2D_common(param, omega_nm, omegaFinal)
                # print(param.shape, omegaFinal)
            temp = d["output"]
            parameter[:, 0] = temp[:, 0]

        # -----------------------------------------------------
        # OPTIONAL  export the interpolated data to txt
        # np.savetxt("intY.txt",parameter,fmt='%1.9e')
        # np.savetxt("distance.txt",distance,fmt='%1.9e')
        # -----------------------------------------------------

        # step 2: generate interpolated parameter for same xaxis as psi
        # and interpolate the parameter to same dimension as psi

        der2 = (0.0, 0.0)
        der2 = fd_ends(distance, parameter)
        secarray2 = spline(distance, parameter, der2[0], der2[1])
        parameter_interp = np.zeros(len(rwave))
        for j in range(0, len(rwave)):
            parameter_interp[j] = splint(distance, parameter, secarray2, rwave[j])

        # step 3: compute the pointwise products
        p1 = np.multiply(psi1, psi2)
        p2 = np.multiply(p1, rwave)
        p3 = np.multiply(p2, rwave)
        product = np.multiply(p3, parameter_interp)

        # function defining the integrand which uses the spline coef array to give interpolated values
        def integrand(xpoint):
            result = splint(rwave, product, secarray2, xpoint)
            return result

        # step 4: gen cubic spline coefs for product
        der2 = fd_ends(rwave, product)
        secarray2 = spline(rwave, product, der2[0], der2[1])

        # step 5: compute the integral using adaptive Quadrature
        result = integrate.quadrature(
            integrand, 0.2, 4.48, tol=1.0e-6, vec_func=False, maxiter=1000
        )
        rounderr = round(result[1], 8)
        if verbose:
            print(
                "{0} < v={1} J={2} | {3} | v={4} J={5} >  =  {6} a.u. (Integration err: {7}) ".format(
                    mol, vl, Jl, name[i], vr, Jr, abs(round(result[0], 7)), rounderr
                )
            )

        if n == 1:
            return abs(round(result[0], 7))


def construct_saveobj(
    vl=np.array([]),
    Jl=np.array([]),
    vr=np.array([]),
    Jr=np.array([]),
    result=np.array([]),
):
    return {
        "vl": vl,
        "Jl": Jl,
        "vr": vr,
        "Jr": Jr,
        "result": result,
    }


def get_filename(mol, wavelength, wavelength_unit, operator):
    wavelength = (
        wavelength if type(wavelength) == str else f"{wavelength:.2f}".replace(".", "")
    )
    return (
        dir_data
        / f"{mol}_{str(wavelength).replace('.', '')}{wavelength_unit}_{operator}.pkl"
    )


def load_from_file(mol, wavelength, wavelength_unit, operator, verbose=False):
    load_path = get_filename(mol, wavelength, wavelength_unit, operator)
    print(load_path)
    if load_path.exists():
        if verbose:
            print(f"Loading {load_path}")
        with open(load_path, "rb") as f:
            obj = pkl.load(f)
    else:
        obj = construct_saveobj()
    return obj


def save_to_file(mol, wavelength, wavelength_unit, operator, obj, verbose=False):
    load_path = get_filename(mol, wavelength, wavelength_unit, operator)
    with open(load_path, "wb") as f:
        if verbose:
            print(f"Saving {load_path}")
        pkl.dump(obj, f)


def load_or_compute(
    mol, vl, Jl, vr, Jr, wavelength, wavelength_unit, operator, obj=None, verbose=False
):
    loaded_obj = obj is None
    if loaded_obj:
        load_from_file(mol, wavelength, wavelength_unit, operator, verbose)

    id = (obj["vl"] == vl) * (obj["Jl"] == Jl) * (obj["vr"] == vr) * (obj["Jr"] == Jr)
    if np.sum(id) == 1:
        return obj["result"][id], obj
    else:
        result_i = compute(
            mol, vl, Jl, vr, Jr, wavelength, wavelength_unit, operator, verbose=verbose
        )
        obj = construct_saveobj(
            *[
                np.append(obj[x], y)
                for x, y in zip(
                    ["vl", "Jl", "vr", "Jr", "result"], [vl, Jl, vr, Jr, result_i]
                )
            ]
        )
        if loaded_obj:
            save_to_file(mol, wavelength, wavelength_unit, operator, obj, verbose)
        return result_i, obj


# ************************************************************************


def compute_batch(
    mol, vl, Jl, vr, Jr, wavelength, wavelength_unit, operator, verbose=False
):
    # This function supports inputting arrays as vl, Jl, vr, Jr. The arrays can have different dimensions.
    # e.g. a 10x1 Jl and 1x4 vl input will result in a 10x4 output.

    # If the inputs are of different dimensions, determine the shape of the resulting matrix
    resultshape = vl * Jl * vr * Jr * np.ones((1,))
    resultshape = np.ones(resultshape.shape)

    # Convert all arguments to 1D numpy arrays
    vl = np.array(vl * resultshape, ndmin=1, dtype=np.int8).flatten()
    Jl = np.array(Jl * resultshape, ndmin=1, dtype=np.int8).flatten()
    vr = np.array(vr * resultshape, ndmin=1, dtype=np.int8).flatten()
    Jr = np.array(Jr * resultshape, ndmin=1, dtype=np.int8).flatten()

    # Compute results for individual quantum numbers
    result = np.zeros((resultshape.size,))
    obj = load_from_file(mol, wavelength, wavelength_unit, operator, verbose)
    for i in range(resultshape.size):
        result[i], obj = load_or_compute(
            mol,
            vl[i],
            Jl[i],
            vr[i],
            Jr[i],
            wavelength,
            wavelength_unit,
            operator,
            obj=obj,
            verbose=verbose,
        )
    save_to_file(mol, wavelength, wavelength_unit, operator, obj, verbose)

    # Reshape into desired shape
    return np.reshape(result, resultshape.shape)


if __name__ == "__main__":
    for vi in range(4):
        print(f"{vi}->{vi+1}")
        x = 0
        for Ji in range(10):
            a = compute("H2", vi + 1, Ji, vi, Ji, 532, "nm", "iso", verbose=False)
            if x == 0:
                x = a
            a = a / x
            print(f"J={Ji:.0f}: a={a:.5f}")
    # print(f'a={compute("H2", 0, 1, 0, 3, 355, "nm", "aniso", verbose=False):.5f}')
    # print(f'a={compute("H2",  0, 0, 1, 0, 632.8, "nm", "iso", verbose=False):.5f}')
