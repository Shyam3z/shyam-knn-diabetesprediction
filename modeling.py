from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import pandas as pd


def data_preprocessing(data):
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Feature scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(data_imputed.drop(["Outcome"], axis=1))
    y = data_imputed['Outcome']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler

def model_building(X_train, X_test, y_train, y_test):
    # Hyperparameter tuning
    param_grid = {'n_neighbors': range(1, 21)}  # K values from 1 to 20
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_k = grid_search.best_params_['n_neighbors']

    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train, y_train)
    return model

def predict_diabetes(model, X_test):
    prediction = model.predict(X_test)
    return prediction

def predict_diabetes_custom(model, scaler, feature_values):
    # Transform the input features using the scaler
    scaled_features = scaler.transform([feature_values])
    # Predict diabetes
    prediction = model.predict(scaled_features)
    return prediction

