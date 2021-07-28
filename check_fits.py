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


for strx in ["10", "1020"]:
    for stry in ["0", "02", "024", "0246", "02468"]:
        str_xy = strx + "_" + stry
        fits_heteroxy = make_fits(np.loadtxt(results_dir
                       / ("gender_age_heteroskedastic_" + str_xy) / "fits.txt"))

        print_stars(f"Improvements gender age heteroskedastic {str_xy}  - homoskedastic")

        print(fits_heteroxy-fits_homo)
