import io
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import date
import logging

logger = logging.getLogger(__name__)

# This method create a linear regression
def lreg(x, y):
    model = LinearRegression().fit(x, y)
    return model, model.score(x, y), model.coef_, model.intercept_

