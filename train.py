import torch
import numpy as np
from datasets import normalization
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# =============================================================================
# Training / validation / test
# =============================================================================
def train_one_epoch(epoch, model, train_loader, optimizer, device, label_coef, lr):
	'''
	Train the neural network for one epoch.
	
	Parameters
	----------
	epoch : int
		Current epoch index. Used only for printing training progress.
	
	model : torch.nn.Module
		Neural network model to be trained. The model should define a
		custom loss function through model.loss().
	
	train_loader : torch.utils.data.DataLoader
		DataLoader that provides mini-batches of training data.
		Each batch contains input features batchX and target labels batchY.
	
	optimizer : torch.optim.Optimizer
		Optimizer used to update model parameters, e.g., Adam or SGD.
	
	device : torch.device
		Computing device, either CPU or GPU.
	
	label_coef : list or tuple
		Normalization parameters for the target variable. Used to convert
		normalized predictions and labels back to the original physical unit.
	
	lr : float
		Current learning rate. Used here for logging/printing only.
	
	Returns
	-------
	targets : np.ndarray
		Training labels after inverse normalization.
	
	preds : np.ndarray
		Model predictions after inverse normalization.
	
	train_r2, train_mae, train_mse, train_rmse, train_mb : float
		Training performance metrics.
	'''
	
	# ------------------------------------------------------------
	# Set the model to training mode.
	# This is important if the model contains layers such as Dropout
	# or BatchNorm, because they behave differently during training
	# and evaluation.
	# ------------------------------------------------------------
	model.train()
	
	# ------------------------------------------------------------
	# Lists used to store true labels and model predictions from
	# all mini-batches. These will be concatenated after the epoch
	# to calculate epoch-level training metrics.
	# ------------------------------------------------------------
	
	targets = []
	preds = []
	
	# ------------------------------------------------------------
	# Loop over all mini-batches in the training DataLoader.
	# Each batch contains:
	#   batchX: input features
	#   batchY: target labels
	# ------------------------------------------------------------
	for batch_index, (batchX, batchY) in enumerate(train_loader):
		# --------------------------------------------------------
		# Move data from CPU memory to the selected computing device.
		# If device is GPU, this allows tensor operations to be
		# accelerated by the GPU.
		# --------------------------------------------------------
		batchX, batchY = batchX.to(device), batchY.to(device)
	
		# --------------------------------------------------------
		# Calculate the training loss for the current mini-batch.
		# The input and label tensors are converted to float because
		# neural network weights are usually stored as floating-point
		# values.
		# --------------------------------------------------------
		# Forward pass: predict ozone from input features
		batchY_hat = model(batchX.float())
		
		# Loss calculation: compare prediction with true ozone label
		loss = model.loss(batchY_hat, batchY.float())
	
		# --------------------------------------------------------
		# Skip this batch if the loss becomes infinite or NaN.
		# This prevents unstable gradients from corrupting the model.
		# However, frequent NaN/Inf losses indicate a deeper issue,
		# such as too large learning rate, bad normalization, or
		# invalid input values.
		# --------------------------------------------------------
		if torch.isinf(loss) or torch.isnan(loss):
			continue
	
		# --------------------------------------------------------
		# Clear old gradients before backpropagation.
		# PyTorch accumulates gradients by default, so this step is
		# required before computing gradients for the current batch.
		# --------------------------------------------------------
		optimizer.zero_grad()
		
		# --------------------------------------------------------
		# Backpropagation.
		# This computes gradients of the loss with respect to all
		# trainable model parameters.
		# ------------------------------------------------------        
		loss.backward()
		
		# --------------------------------------------------------
		# Update model parameters using the gradients computed above.
		# For example, if optimizer is Adam, this applies one Adam
		# update step.
		# --------------------------------------------------------
		optimizer.step()
	
		# --------------------------------------------------------
		# Generate predictions for the current batch.
		# detach() removes the tensor from the computation graph so
		# gradients are not tracked.
		# cpu() moves the tensor back to CPU memory.
		# numpy() converts the tensor to a NumPy array for metric
		# calculation later.
		#
		# Note: these predictions are made after optimizer.step(),
		# so they reflect the updated model parameters.
		# --------------------------------------------------------
		predictions = model(batchX.float()).detach().cpu().numpy()
		
		# --------------------------------------------------------
		# Store predictions and true labels for epoch-level metrics.
		# batchY is also moved back to CPU before converting to NumPy.
		# --------------------------------------------------------    
		preds.append(predictions)
		targets.append(batchY.cpu().numpy())
		
	# ------------------------------------------------------------
	# Concatenate mini-batch results into one continuous array.
	# reshape((-1,)) converts the output to a 1D array, which is the
	# expected format for most regression metric functions.
	# ------------------------------------------------------------
	targets = np.concatenate(targets, axis=0).reshape((-1,))
	preds = np.concatenate(preds, axis=0).reshape((-1,))
	
	# ------------------------------------------------------------
	# Convert labels and predictions from normalized space back to
	# the original physical unit.
	#
	# For the ozone example, this means converting predictions back
	# to the original ozone unit, such as ppb.
	# ------------------------------------------------------------
	targets, _ = normalization(targets, label_coef, "inverse")
	preds, _ = normalization(preds, label_coef, "inverse")
	
	# ------------------------------------------------------------
	# Calculate epoch-level training metrics.
	# ------------------------------------------------------------
	train_r2, train_mae, train_mse, train_rmse, train_mb = calc_metrics(targets, preds)
	
	print(
		f"Train: {epoch + 1} LR: {lr:.5f}, "
		f"r2_scores: {train_r2:.2f} mae_scores: {train_mae:.3f} "
		f"mse_scores: {train_mse:.3f} rmse_scores: {train_rmse:.3f} "
		f"mb_scores: {train_mb:.3f}"
	)
	
	return targets, preds, train_r2, train_mae, train_mse, train_rmse, train_mb

def validation(epoch, model, validation_loader, device, label_coef, lr):
    model.eval()

    with torch.no_grad():
        targets = []
        preds = []

        for index, (batchX, batchY) in enumerate(validation_loader):
            batchX, batchY = batchX.to(device), batchY.to(device)

            predictions = model(batchX.float()).cpu().numpy()

            preds.append(predictions)
            targets.append(batchY.cpu().numpy())

        targets = np.concatenate(targets, axis=0).reshape((-1,))
        preds = np.concatenate(preds, axis=0).reshape((-1,))

        targets, _ = normalization(targets, label_coef, "inverse")
        preds, _ = normalization(preds, label_coef, "inverse")

        valid_r2, valid_mae, valid_mse, valid_rmse, valid_mb = calc_metrics(targets, preds)

        print(
            f"Evaluate: {epoch + 1} LR: {lr:.5f}, "
            f"r2_scores: {valid_r2:.3f} mae_scores: {valid_mae:.3f} "
            f"mse_scores: {valid_mse:.3f} rmse_scores: {valid_rmse:.3f} "
            f"mb_scores: {valid_mb:.3f}"
        )

    return targets, preds, valid_r2, valid_mae, valid_mse, valid_rmse, valid_mb


def test(model, data_test, device, label_coef):
    model.eval()

    with torch.no_grad():
        batchX = data_test.x.to(device)
        batchY = data_test.y.to(device)
        predictions = model(batchX.float()).cpu().numpy()

    targets = batchY.cpu().numpy().reshape((-1,))
    preds = predictions.reshape((-1,))

    targets, _ = normalization(targets, label_coef, "inverse")
    preds, _ = normalization(preds, label_coef, "inverse")

    test_r2, test_mae, test_mse, test_rmse, test_mb = calc_metrics(targets, preds)

    print("=" * 90)
    print(
        f"Test, r2_scores: {test_r2:.2f} mae_scores: {test_mae:.3f} "
        f"mse_scores: {test_mse:.3f} rmse_scores: {test_rmse:.3f}"
        f"mb_scores: {test_mb:.3f}"
    )

    return targets, preds, test_r2, test_mae, test_mse, test_rmse, test_mb


def calc_metrics(targets, preds):
    targets = np.asarray(targets).reshape(-1)
    preds = np.asarray(preds).reshape(-1)

    idx = np.isfinite(targets) & np.isfinite(preds)
    if idx.sum() == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    targets_valid = targets[idx]
    preds_valid = preds[idx]

    r2 = r2_score(targets_valid, preds_valid)
    mae = mean_absolute_error(targets_valid, preds_valid)
    mse = mean_squared_error(targets_valid, preds_valid)
    rmse = np.sqrt(mse)
    mb = np.mean(preds_valid - targets_valid)

    return r2, mae, mse, rmse, mb
    
    
    
