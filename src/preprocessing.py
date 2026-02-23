# TODO: Implement preprocessing functions
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def preprocess_data(X, categorical_features, numerical_features):
    """
    Preprocess the data by applying scaling to numerical features and encoding to categorical features.

    Parameters:
    X (pd.DataFrame): The input data to be preprocessed.
    categorical_features (list): A list of column names corresponding to categorical features.
    numerical_features (list): A list of column names corresponding to numerical features.

    Returns:
    np.ndarray: The preprocessed data ready for model training or prediction.
    """
    # Define the transformers for numerical and categorical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine transformers into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create a pipeline that applies the preprocessor
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    # Fit and transform the data
    X_preprocessed = pipeline.fit_transform(X)

    return X_preprocessed