import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess_data(path='data/cleaned_lifestyle_data_with_BMI.csv'):
    df = pd.read_csv(path)

    # Rename columns to consistent naming used across app and EDA
    df.rename(columns={
        'RIAGENDR': 'Gender',
        'RIDAGEYR': 'Age',
        'BMXWT': 'Weight',
        'BMXHT': 'Height',
        'BMXWAIST': 'Waist',
        'BPXSY1': 'Systolic BP',
        'BPXDI1': 'Diastolic BP'
    }, inplace=True)

    # Handle Gender mapping
    if df['Gender'].dtype in [int, float]:
        df['Gender'] = df['Gender'].map({1: 'Male', 2: 'Female'})

    # Drop invalid rows
    df.dropna(subset=['BMI', 'Age'], inplace=True)

    # Encode Gender
    le = LabelEncoder()
    df['Gender_encoded'] = le.fit_transform(df['Gender'])

    # Define feature matrix and target
    X = df[['Age', 'Gender_encoded', 'Weight', 'Height', 'Waist']]
    y = df['BMI']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return df, X_train, X_test, y_train, y_test
