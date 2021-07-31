from pathlib import Path
import numpy as np

from cupid_utils import N_HOUSEHOLDS_OBS, print_stars

def make_fits(fits):
    corrected_fits = np.zeros(3)
    corrected_fits[0] = fits[0]*N_HOUSEHOLDS_OBS 
    corrected_fits[1:] = -fits[1:]
    return corrected_fits

results_dir = Path("./Results")

fits_homo = make_fits(np.loadtxt(results_dir
                       / "homoskedastic/fits.txt"))


fits_hetero = make_fits(np.loadtxt(results_dir
                       / "gender_heteroskedastic/fits.txt"))

print_stars("Improvements gender heteroskedastic - homoskedastic")

print(fits_hetero-fits_homo)


strx = "1020"
stry = "0"

str_xy = strx + "_" + stry
fits_heteroxy = make_fits(np.loadtxt(
    results_dir / ("gender_age_heteroskedastic_" + str_xy) / "fits.txt"))

print_stars(f"Improvements gender age heteroskedastic {str_xy}  - homoskedastic")

print(fits_heteroxy-fits_homo)

fits_fcmnl0 = make_fits(np.loadtxt(results_dir / f"Fcmnl_b0" / "fits.txt"))

print_stars("Fcmnl_b0  - homoskedastic is close to zero as expected")

print(fits_fcmnl0-fits_homo)


for b_case in ["_interp3", "_interp4", "_interp3_0", "_interp3_0_5"]:
    fits_fcmnl = make_fits(np.loadtxt(
        results_dir / f"Fcmnl_b{b_case}" / "fits.txt"))

    print_stars(f"Improvements Fcmnl_b{b_case}  - Fcmnl_b0")

    print(fits_fcmnl-fits_fcmnl0)
