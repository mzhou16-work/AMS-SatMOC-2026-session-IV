import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# ----------------------------------------------------------------------
class OzoneDataset(Dataset):
    def __init__(self, label, feature):
        self.y = torch.from_numpy(label.astype(np.float32))
        self.x = torch.from_numpy(feature.astype(np.float32))

        print(f" - Shape of the label: {self.y.shape}")
        print(f" - Shape of the features: {self.x.shape}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# ======================================================================
# Data manipulator 
# ======================================================================

def normalization(data, p=None, operation="forward", eps=1e-12):
    """
    Gaussian normalization:
        x_norm = (x - mean) / std

    Parameters
    ----------
    data : np.ndarray
        1D or 2D array.
    p : tuple/list or None
        Normalization coefficients. For 1D: (mean, std).
        For 2D: list of (mean, std), one tuple per column.
    operation : {"forward", "inverse"}
    eps : float
        Minimum std used to avoid division by zero.
    """
    data = np.asarray(data, dtype=np.float64)

    if operation == "forward":
        if data.ndim == 1:
            mu = np.nanmean(data)
            sig = np.nanstd(data)
            sig = sig if sig > eps else 1.0
            x_ = (data - mu) / sig
            p = (mu, sig)

        elif data.ndim == 2:
            x_ = np.full_like(data, np.nan, dtype=np.float64)
            p = []
            for i in range(data.shape[1]):
                mu = np.nanmean(data[:, i])
                sig = np.nanstd(data[:, i])
                sig = sig if sig > eps else 1.0
                x_[:, i] = (data[:, i] - mu) / sig
                p.append((mu, sig))
        else:
            raise ValueError("normalization only supports 1D or 2D arrays")

    elif operation == "inverse":
        if data.ndim == 1:
            if isinstance(p, list) and len(p) == 1:
                p = p[0]
            mu, sig = p
            x_ = data * sig + mu

        elif data.ndim == 2:
            # For a single-output label with shape [N, 1], p may be a single tuple.
            if isinstance(p, tuple) and len(p) == 2:
                x_ = data * p[1] + p[0]
            else:
                x_ = np.full_like(data, np.nan, dtype=np.float64)
                for i in range(data.shape[1]):
                    mu, sig = p[i]
                    x_[:, i] = data[:, i] * sig + mu
        else:
            raise ValueError("normalization only supports 1D or 2D arrays")

    else:
        raise ValueError("operation must be 'forward' or 'inverse'")

    return x_, p

def split_data(label, feat, validation_split, test_split, random_seed=42, shuffle_dataset=True):
    dataset_size = len(label)
    print(f" - Receive {dataset_size} data")
    print(
        f" - Split {(1-validation_split-test_split)*100:.1f}% to training, "
        f"{validation_split*100:.1f}% to validation, {test_split*100:.1f}% to test"
    )

    indices = list(range(dataset_size))

    train_split_idx = int(np.floor((1 - validation_split - test_split) * dataset_size))
    validation_split_idx = int(np.floor((1 - test_split) * dataset_size))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices = indices[:train_split_idx]
    val_indices = indices[train_split_idx:validation_split_idx]
    test_indices = indices[validation_split_idx:]

    print(
        f" - training: {len(train_indices)}, "
        f"validation: {len(val_indices)}, test: {len(test_indices)}"
    )
    return train_indices, val_indices, test_indices


# ======================================================================
# Ozone data reader
# ======================================================================
def read_ozone_data(
    filename,
    read_vars,
    label_name,
    feat_name,
    categorical_feat_name=None,
    ozone_valid_range=None,
):

	'''
	Read ozone training data from file and prepare labels and features
	for machine-learning model training.
	
	This function is responsible for loading the raw ozone dataset,
	selecting the target variable and predictor variables, applying
	optional quality control to ozone observations, and organizing the
	data into model-ready arrays.
	
	Parameters
	----------
	filename : str
		Path to the input data file. The file contains ozone labels and
		predictor variables, such as meteorology, chemistry, land-surface,
		temporal, and spatial features.
	
	read_vars : list of str
		List of variables to read from the input file. This usually includes
		both the target variable and all candidate predictor variables.
		Reading only selected variables can reduce memory use and improve
		data-loading speed.
	
	label_name : str
		Name of the target variable to predict.
		For the ozone demo, this is likely the observed surface ozone variable,
		such as EPA ozone concentration.
	
	feat_name : list of str
		Names of continuous or numerical predictor variables used as model
		inputs. Examples may include latitude, longitude, temperature,
		boundary-layer height, chemical fields, NDVI, population, day of year,
		and other environmental predictors.
	
	categorical_feat_name : list of str, optional
		Names of categorical predictor variables, if any.
		These variables may need special processing, such as one-hot encoding
		or embedding, before being used by machine-learning models.
		If None, no categorical features are used.
	
	ozone_valid_range : tuple or list, optional
		Valid range used to filter ozone labels, for example (0, 150).
		Samples with ozone values outside this range are removed.
		This helps exclude physically unrealistic values, missing values,
		or poor-quality observations.
	
	Returns
	-------
	label : np.ndarray
		Target ozone values after quality control and filtering.
	
	feat : np.ndarray
		Predictor-feature matrix after variable selection and preprocessing.
		The expected shape is usually:
	
			(number_of_samples, number_of_features)
	
	optional outputs
		Depending on the implementation, the function may also return
		feature names, categorical encoders, normalization parameters,
		or quality-control masks.
	'''
	
	print(f" - Reading {filename}")
	
	if categorical_feat_name is None:
		categorical_feat_name = []
	
	df = pd.read_csv(filename)
	
	required_vars = list(read_vars)
	missing_cols = [c for c in required_vars if c not in df.columns]
	if len(missing_cols) > 0:
		raise ValueError(f"Missing columns in input CSV: {missing_cols}")
	
	df = df[required_vars]
	df = df.replace([np.inf, -np.inf], np.nan)
	df = df.dropna()
	df = df.drop_duplicates()
	
	# --------------------------------------------------------------
	# Ozone physical-range filtering
	# --------------------------------------------------------------
	if ozone_valid_range is not None:
		o3_min, o3_max = ozone_valid_range
		y_col = label_name[0]
		df = df[(df[y_col] >= o3_min) & (df[y_col] <= o3_max)]
	
	# --------------------------------------------------------------
	# Keep continuous features, categorical features, and labels
	# --------------------------------------------------------------
	all_feat_name = feat_name + categorical_feat_name
	new_df = df[all_feat_name + label_name].copy()
	
	# Negative longitude, wind, anomalies, and standardized quantities are valid.
	# Only remove NaN / Inf values.
	valid_id = np.ones(len(new_df), dtype=bool)
	for var in all_feat_name + label_name:
		valid_id &= np.isfinite(new_df[var].values)
	
	new_df = new_df.loc[valid_id].copy()
	
	# --------------------------------------------------------------
	# Normalize label and continuous features
	# --------------------------------------------------------------
	trans_label, coef_label = normalization(new_df[label_name].values)
	trans_feat_cont, coef_feat = normalization(new_df[feat_name].values)
	
	# --------------------------------------------------------------
	# One-hot encode categorical features
	# --------------------------------------------------------------
	onehot_blocks = []
	onehot_feature_names = []
	categorical_dict = {}
	
	for cat_name in categorical_feat_name:
		cat_values = new_df[cat_name]
	
		# For land-type classes stored as float, convert 1.0, 2.0, ... to int.
		if np.issubdtype(cat_values.dtype, np.number):
			cat_values = cat_values.astype(int)
	
		onehot_df = pd.get_dummies(
			cat_values,
			prefix=cat_name,
			dtype=np.float32,
		)
	
		# Stable column order.
		onehot_df = onehot_df.reindex(sorted(onehot_df.columns), axis=1)
	
		onehot_blocks.append(onehot_df.values.astype(np.float32))
		onehot_feature_names.extend(onehot_df.columns.tolist())
	
		categorical_dict[cat_name] = {
			"type": "one_hot",
			"columns": onehot_df.columns.tolist(),
		}
	
	# --------------------------------------------------------------
	# Concatenate continuous + categorical features
	# --------------------------------------------------------------
	if len(onehot_blocks) > 0:
		trans_feat_cat = np.concatenate(onehot_blocks, axis=1)
		trans_feat = np.concatenate([trans_feat_cont, trans_feat_cat], axis=1)
	else:
		trans_feat = trans_feat_cont
	
	# --------------------------------------------------------------
	# Save normalization coefficients
	# --------------------------------------------------------------
	label_dict = {}
	feature_dict = {}
	
	for name, coef in zip(label_name, coef_label):
		label_dict[name] = coef
	
	for name, coef in zip(feat_name, coef_feat):
		feature_dict[name] = coef
	
	for name, info in categorical_dict.items():
		feature_dict[name] = info
	
	order_feat = feat_name + onehot_feature_names
	
	# --------------------------------------------------------------
	# Print data summary
	# --------------------------------------------------------------
	print(" - Feature configuration")
	print(f"   Number of continuous features : {len(feat_name)}")
	print(f"   Number of categorical features: {len(categorical_feat_name)}")
	print(f"   Number of one-hot features    : {len(onehot_feature_names)}")
	print(f"   Total model input features    : {len(order_feat)}")
	print("   Feature order:")
	for i, name in enumerate(order_feat, start=1):
		print(f"      {i:02d}. {name}")
	print(f"   Feature array shape           : {trans_feat.shape}")
	print()
	
	print(" - Label configuration")
	print(f"   Number of labels              : {len(label_name)}")
	print(f"   Label order                   : {', '.join(label_name)}")
	print(f"   Label array shape             : {trans_label.shape}")
	
	return trans_label, trans_feat, label_dict, feature_dict, order_feat


