import pandas as pd
import numpy as np
import sys
import pdb

csv_file = sys.argv[1]
data = pd.read_csv(csv_file, low_memory=False)
df = pd.DataFrame(data)
max_list = []
for i in range(10):
	start = i * 100 + i * 2
	end = (i + 1) * 100 + i * 2 + 1
	max_list.append(np.max(np.array(df['auc_prot_med_test'][start:end], dtype=np.float32)))
print(max_list)
print("mean is {:}".format(np.mean(np.array(max_list))))
