import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from keras.models import Sequential, Model
from keras.layers import Conv1D, LSTM, Dense, Dropout, Input, MaxPooling1D, Flatten, Reshape, ConvLSTM2D
from keras.regularizers import l2
from keras.callbacks import EarlyStopping


def train_validate(
    splits,
    lookback,
    horizon,
    model_builder,
    scale_function,
    sequence_creator,
    epochs,
    batch_size,
    verbose=1,
    patience=5
):
    """
    General training and validation pipeline for various model architectures.

    Parameters:
        splits (list): List of (train_df, val_df) tuples.
        lookback (int): Lookback window size.
        horizon (int): Prediction horizon.
        model_builder (function): Function to build the desired model architecture.
        scale_function (function): Function to scale data, returns scalers and scaled data.
        sequence_creator (function): Function to create input sequences.
        epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
        verbose (int): Verbosity level for training logs.
        patience (int): Early stopping patience.

    Returns:
        dict: Summary results.
        list: Split-wise results.
        Model: Trained model from the last split.
    """
    results = []
    total_training_time = 0
    final_model = None

    for i, (train_df, val_df) in enumerate(splits):
        print(f"\nProcessing Split {i + 1}/{len(splits)}")

        # Scale data
        scaled_train, scaled_val, scalers = scale_function(train_df, val_df)

        # Create sequences
        X_train, y_train, _, _ = sequence_creator(scaled_train, lookback, horizon)
        X_val, y_val, val_anomalies, _ = sequence_creator(scaled_val, lookback, horizon)

        # Adjust input shape if necessary
        if model_builder.__name__ in ["build_convlstm"]:
            X_train = X_train[..., np.newaxis]
            X_val = X_val[..., np.newaxis]

        # Build the model
        n_features = X_train.shape[2]
        model = model_builder(lookback, horizon, n_features)

        # Early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        # Train the model
        start_time = time.time()
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=verbose,
        )
        split_training_time = time.time() - start_time
        total_training_time += split_training_time

        # Evaluate model
        y_pred = model.predict(X_val)
        y_val_og = scalers["target"].inverse_transform(y_val)
        y_pred_og = scalers["target"].inverse_transform(y_pred)

        mae = mean_absolute_error(y_val_og, y_pred_og)
        rmse = np.sqrt(mean_squared_error(y_val_og, y_pred_og))
        anom_mae, anom_rmse = [], []

        for step in range(horizon):
            step_anomaly_mask = val_anomalies[:, step] == 1
            if np.any(step_anomaly_mask):
                anom_mae.append(
                    mean_absolute_error(
                        y_val_og[step_anomaly_mask, step], y_pred_og[step_anomaly_mask, step]
                    )
                )
                anom_rmse.append(
                    np.sqrt(
                        mean_squared_error(
                            y_val_og[step_anomaly_mask, step], y_pred_og[step_anomaly_mask, step]
                        )
                    )
                )
            else:
                anom_mae.append(np.nan)
                anom_rmse.append(np.nan)

        results.append({
            'split': i + 1,
            'Overall_MAE': mae,
            'Overall_RMSE': rmse,
            'Anom_MAE': np.nanmean(anom_mae),
            'Anom_RMSE': np.nanmean(anom_rmse),
        })

        print(
            f"Split {i + 1} - Overall MAE: {mae:.4f}, Overall RMSE: {rmse:.4f}, "
            f"Overall Anomaly MAE: {np.nanmean(anom_mae):.4f}, Overall Anomaly RMSE: {np.nanmean(anom_rmse):.4f}"
        )

        final_model = model

    # Aggregate results
    avg_overall_mae = np.mean([res['Overall_MAE'] for res in results])
    avg_overall_rmse = np.mean([res['Overall_RMSE'] for res in results])
    avg_overall_anom_mae = np.nanmean([res['Anom_MAE'] for res in results])
    avg_overall_anom_rmse = np.nanmean([res['Anom_RMSE'] for res in results])

    minutes, seconds = divmod(total_training_time, 60)
    summary_results = {
        'Average Overall MAE': avg_overall_mae,
        'Average Overall RMSE': avg_overall_rmse,
        'Average Overall Anomaly MAE': avg_overall_anom_mae,
        'Average Overall Anomaly RMSE': avg_overall_anom_rmse,
        'Total Training Time': f"{int(minutes)}m {int(seconds)}s",
    }

    return summary_results, results, final_model
