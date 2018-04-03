# Project Title

This is a repository of ad-hoc scripts produced as part of a research project
investigating how to create synthetic data for fairness aware recommender
systems.  There are two directories: `analysis` and `distribution`.  Both
directories have a `data` directory.  The `data` directory should be
symlinked to a directory containing the XING dataset.

## Distribution Directory

The distribution directory contains scripts for analyzing the XING dataset,
fitting distributions to the XING dataset, and creating synthetic distributions
that represent protected groups.

### distribution_analysis

This notebook looks at the click propensity distribution in the XING dataset and
fits a number of mathematical distributions to the dataset using the `fitter`
library.  The best fitting mathematical distributions are plotted against the
the original distribution for visual comparison.

### log powerlaw distribution

The log-powerlaw distribution is compared to the log-transformed dataset.
Lots of math in here, but in the end this work is not used because once the
log tranformed distribution is transformed back into linear space, the
synthetically generated distributions are not useful.

### piecewise distribution

The underlying procedure for generating a synthetic piecewise distribution is
located here.  This includes some analysis of the relationship between the
distribution variable lambda and the overall expectation that a user will belong
to the protected group.

### probability distribution

This contains the procedure or generating 'ratio' based synthetic distributions.
In the ratio based scheme, the ratio of users in group A vs. group B linearly
falls to zero as click propensity increases.  The initial ratio is an adjustable
parameter.

## Analysis Directory

This directory contains all work done after the synthetic distributions are
generated.

### delta_7_7_analysis

Contains analysis of the piecewise synthetic distribution with a lambda of 7.
