#HANDLING MISSING VALUES

import pandas as pd
import numpy as np

sf_permits = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv")
np.random.seed(0) 

sf_permits.head()

missing = sf_permits.isnull().sum()
cells = np.product(sf_permits.shape)
total_missing = missing.sum()
percent_missing = (total_missing/cells) * 100
print(percent_missing)

df = sf_permits
df.dropna()

sf_permits_with_na_dropped = sf_permits.dropna(axis=1)
print("Columns with na's dropped: %d" % sf_permits_with_na_dropped.shape[1])
print("Columns in original dataset: %d \n" % sf_permits.shape[1])

dropped_columns = 31
sf_permits_with_na_imputed = sf_permits.fillna(method='bfill', axis=0).fillna(0)
