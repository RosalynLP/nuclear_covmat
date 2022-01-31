# nuclear_covmat
Code for computing nuclear covariance matrix for inclusion in NNPDF4.0 fits.

- Plots are outputted in the plots folder
- Code can be found in src

## Structure of src
### observables
- This contains the validphys runcards for use with [nnpdf code](github.com/NNPDF/nnpdf) to get nuclear and proton observables
### covmat
- This contains the script covmat.py which calculates the nuclear covmat

## Workflow
1. Run all the runcards in src/observables. This requires a working version of the [nnpdf code](github.com/NNPDF/nnpdf).
2. Run src/covmat/covmat.py to get the nuclear covariance matrix.

## Useful links
1. [deuteron covariance matrix project](github.com/RosalynLP/deuteron_corrections).
2. [Summary conference proceedings](https://arxiv.org/abs/2106.12349).
3. [NNPDF github](github.com/NNPDF).
