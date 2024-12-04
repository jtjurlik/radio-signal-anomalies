from sklearn.preprocessing import MinMaxScaler, StandardScaler


def scale_data_split(train_df, val_df, temporal_features, static_features):
    scaler_temporal = StandardScaler()
    scaler_static = MinMaxScaler()
    scaler_target = StandardScaler()

    # Scale time-variant features
    if temporal_features:
        train_df[temporal_features] = scaler_temporal.fit_transform(train_df[temporal_features])
        val_df[temporal_features] = scaler_temporal.transform(val_df[temporal_features])

    # Scale time-invariant features
    if static_features:
        train_df[static_features] = scaler_static.fit_transform(train_df[static_features])
        val_df[static_features] = scaler_static.transform(val_df[static_features])

    # Scale minRSSI separately (target variable)
    train_df['minRSSI'] = scaler_target.fit_transform(train_df[['minRSSI']])
    val_df['minRSSI'] = scaler_target.transform(val_df[['minRSSI']])

    return train_df, val_df, scaler_target, scaler_temporal, scaler_static


def scale_test_data(test_df, scaler_target, scaler_temporal, scaler_static, temporal_features, static_features):
    """Scale the test data using the same scalers as for the training/validation set."""
    # Scale minRSSI separately (target variable)
    test_df['minRSSI'] = scaler_target.transform(test_df[['minRSSI']])

    if temporal_features:
      test_df[temporal_features] = scaler_temporal.transform(test_df[temporal_features])

    if static_features:
      test_df[static_features] = scaler_static.transform(test_df[static_features])

    return test_df
