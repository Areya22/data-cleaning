#CHARACTER ENCODINGS
import pandas as pd
import numpy as np
import chardet

np.random.seed(0)
sample_entry = b'\xa7A\xa6n'
print(sample_entry)
print('data type:', type(sample_entry))
df = sample_entry.decode("big5-tw")
new_entry = df.encode("utf-8", errors="replace")
police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv")
police_killings.to_csv("/kaggle/working/my_file.csv")