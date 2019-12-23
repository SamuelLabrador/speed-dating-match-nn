import pandas as pd
import numpy as np

# Normalization
from sklearn import preprocessing

# Load Data
file = 'data/raw_data.csv'
fh = open(file, 'r')

# Load 
raw_data = pd.read_csv(fh)
data = raw_data.copy()

# Get rid of na values in data
data = data.dropna()
print('data shape:', data.shape)
skip_list = ['d_age',]
one_hot = ['race', 'race_o', 'field']

# Clean data
# One hot encoding for ordinary features
# and integer encoding for d_ ranges
for value in data.columns:
	if value in skip_list:
		continue

	if value in one_hot:
		oh = pd.get_dummies(data[value], prefix=value)
		data = data.drop(value, axis=1)
		data = data.join(oh)

	if 'd_' == value[0:2] or value == 'gender':
		values = np.unique(data[value].T)
		
		# Generate mapping
		mapping = dict()
		for i, tag in enumerate(values):
			mapping[tag] = int(i)
		
		# Apply mapping to column		
		data[value] = data[value].apply(lambda x : mapping[x])

# Normalize Data
for value in data.columns:
	x = data[[value]].values.astype(float)
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(x)

	data[value] = x_scaled

# Get rid of irrelevant field
data = data.drop('has_null', axis=1)

# Load into tensorflow dataset
data = data.sample(frac=1).reset_index(drop=True)
out_file = 'data/cleaned_data.csv'
data.to_csv(out_file)
