import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
data = pd.read_csv("dataset.csv")
imputer = SimpleImputer(strategy="mean")
data_imputed = imputer.fit_transform(data)
encoder = OneHotEncoder()
categorical_features = ["categorical_column"]
encoded_categorical = encoder.fit_transform(data[categorical_features])
scaler = StandardScaler()
numerical_features = ["numerical_column"]
scaled_numerical = scaler.fit_transform(data[numerical_features])
preprocessed_features = pd.concat([pd.DataFrame(scaled_numerical), pd.DataFrame(encoded_categorical.toarray())], axis=1)
preprocessing_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="mean")),
    ('encoder', OneHotEncoder()),
    ('scaler', StandardScaler())
])
preprocessed_data = preprocessing_pipeline.fit_transform(data)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

preprocessed_data = preprocessor.fit_transform(data)
