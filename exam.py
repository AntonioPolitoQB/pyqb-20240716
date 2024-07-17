# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Programming in Python
# ## Exam: July 16, 2024
#
# You can solve the exercises below by using standard Python 3.11 libraries, NumPy, Matplotlib, Pandas, PyMC.
# You can browse the documentation: [Python](https://docs.python.org/3.11/), [NumPy](https://numpy.org/doc/1.26/index.html), [Matplotlib](https://matplotlib.org/3.8.2/users/index.html), [Pandas](https://pandas.pydata.org/pandas-docs/version/2.1/index.html), [PyMC](https://www.pymc.io/projects/docs/en/v5.10.3/api.html).
# You can also look at the slides or your code on [GitHub](https://github.com). 
#
# **It is forbidden to communicate with others or "ask questions" online (i.e., stackoverflow is ok if the answer is already there, but you cannot ask a new question or use ChatGPT or similar products)**
#
# To test examples in docstrings use
#
# ```python
# import doctest
# doctest.testmod()
# ```
#

import numpy as np
import pandas as pd             # type: ignore
import matplotlib.pyplot as plt # type: ignore
import pymc as pm               # type: ignore
import arviz as az              # type: ignore

# ### Exercise 1 (max 3 points)
#
# The file [brown_bear_blood.csv](./brown_bear_blood.csv) (Shimozuru, Michito, Nakamura, Shiori, Yamazaki, Jumpei, Matsumoto, Naoya, Inoue-Murayama, Miho, Qi, Huiyuan, Yamanaka, Masami, Nakanishi, Masanao, Yanagawa, Yojiro, Sashika, Mariko, Tsubota, Toshio, & Ito, Hideyuki. (2023). *Age estimation based on blood DNA methylation levels in brown bears* https://doi.org/10.5061/dryad.9w0vt4bm0) contains
#
#  - Bear ID
#  - Birth date. It was assumed that all bears were born on February 1.
#  - Date of the blood sampling.
#  - Ages were determined at the time of blood sampling
#  - Sex, F: female, M: male.
#  - Growth environment (i.e., captive or wild).
#  - Values of the methylation levels of the samples. As PCR for each sample was conducted in duplicate, the average  value was taken as the methylation level for each sample. 
#
# Load the data in a Pandas dataframe. Be sure the columns with dates have the correct dtype (`datetime64[ns]`) and the dates are parsed correctly (the birth date is always on February 1).

pass

# ### Exercise 2 (max 4 points)
#
# Add a column `sampling_place` with the last word of `Sample_ID`.
#

pass

# ### Exercise 3 (max 7 points)
#
# Define a function `oddity` that takes a Pandas Series of numerical values and returns a new Series in which every element is multiplied according to its integer index: if the index is even the original value is multiplied by 100, if the index is odd the original value is multiplied by 1000. 
#
# For example, if the Series contains the values 10., 9.5, 8., then the result is 1000., 9500, 800. 
#
#
# To get the full marks, you should declare correctly the type hints and add a test within a doctest string.

pass


# +
# You can test your docstrings by uncommenting the following two lines

# import doctest
# doctest.testmod()
# -

# ### Exercise 4 (max 5 points)
#
# Apply the function defined in Exercise 3 on the two series of `age_years` of male and female bears, each ordered by increasing `age_years`. Avoid the use of loops for full marks.

pass

# ### Exercise 5 (max 3 points)
#
# Print all the unique names of the `sampling_place` together with the number of samples collected in that site.

pass

# ### Exercise 6 (max 3 points)
#
# Plot together the histograms of `age_years` for each combination of sex and environment. The four histograms should appear within the same axes.

pass


# ### Exercise 7 (max 4 points)
#
# Make a figure with 3 rows with the scatter plots of `age_years` vs. the three methylation levels (`SLC12A5`,`VGF`,`SCGN`).

# +
pass



# -

# ### Exercise 8 (max 4 points)
#
# Consider this statistical model:
#
# - a parameter $\alpha$ is normally distributed with $\mu = 0$ and $\sigma = 1$ 
# - a parameter $\beta$ is normally distributed with $\mu = 1$ and $\sigma = 1$ 
# - a parameter $\gamma$ is exponentially distributed with $\lambda = 1$
# - the observed `age_years` is normally distributed with standard deviation $\gamma$ and a mean given by $\alpha + \beta \cdot M$ (where $M$ is the correspondig value of `SCGN`).
#
# Code this model with pymc, sample the model, and plot the summary of the resulting estimation by using `az.plot_posterior`. 
#
#
#

pass
