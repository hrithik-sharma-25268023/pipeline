"""module for model inputs"""

import os
import pickle
import pandas as pd


if __name__ == "__main__":
    path = os.getcwd()+"/data/income.csv"
    data = pd.read_csv(path).head()
    print(data)
    print(data.columns)
