import numpy as np
import pandas as pd

from rpy2.robjects import r, globalenv
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import Converter

# Build converter
converter = Converter('pandas_converter')
converter = converter + pandas2ri.converter

def variance_explained_mixomics(X_df, Y_df, ncomp):
    """
    mixOmics-style explained variance using rpy2 (modern API).
    """

    if isinstance(Y_df, pd.Series):
        Y_df = Y_df.to_frame()

    with converter.context():
        # Assign directly to R global environment
        globalenv["Xr"] = X_df
        globalenv["Yr"] = Y_df
        globalenv["ncomp"] = ncomp

        r("""
        suppressMessages(library(mixOmics))

        pls_model <- pls(Xr, Yr, ncomp = ncomp)

        explX <- pls_model$explained_variance$X
        explY <- pls_model$explained_variance$Y

        cum_explX <- cumsum(explX)
        cum_explY <- cumsum(explY)
        """)

        X_var = np.array(r["explX"], dtype=float)
        Y_var = np.array(r["explY"], dtype=float)
        cum_X = np.array(r["cum_explX"], dtype=float)
        cum_Y = np.array(r["cum_explY"], dtype=float)

    return {
        "X_var": X_var,
        "Y_var": Y_var,
        "cum_X": cum_X,
        "cum_Y": cum_Y
    }
