import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle

# Load your dataset
data_path = '../data/processed/merged_data.csv'
data = pd.read_csv(data_path, low_memory=False)

# Create a custom transformer for date features
class DateTransformer:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['Date'] = pd.to_datetime(X['Date'])
        X['Year'] = X['Date'].dt.year
        X['Month'] = X['Date'].dt.month
        X['Week'] = X['Date'].dt.isocalendar().week
        X['Day'] = X['Date'].dt.day
        X['DayOfYear'] = X['Date'].dt.dayofyear
        X['Weekday'] = X['Date'].dt.weekday  # Monday=0, Sunday=6
        X['Weekend'] = (X['Weekday'] >= 5).astype(int)  # 1 if Saturday/Sunday, else 0

        return X.drop(columns=['Date'])

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

# Transform the entire dataset before splitting
data = DateTransformer().fit_transform(data)

# Define features and target variable
X = data.drop(columns=['Sales'])  # Assuming 'Sales' is your target variable
y = data['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define numeric and categorical features
numeric_features = [
    'Customers',
    'CompetitionDistance',
    'CompetitionOpenSinceMonth',
    'CompetitionOpenSinceYear',
    'Promo2SinceWeek',
    'Promo2SinceYear',
    'Year',
    'Month',
    'Week',
    'Day',
    'DayOfYear',
    'Weekday',
    'Weekend',
]

categorical_features = [
    'DayOfWeek',
    'StateHoliday',
    'StoreType',
    'Assortment',
    'PromoInterval',
]

# Convert categorical features to string AFTER the split
for col in categorical_features:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)

# Create the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )),
    ('model', RandomForestRegressor(n_estimators=100,max_depth=10, n_jobs=-1, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Save the trained pipeline
with open('pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("Model trained and saved!")