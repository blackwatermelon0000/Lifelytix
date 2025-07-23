import pandas as pd
import os

# === Set data path ===
data_path = r'C:\Users\Ayesha Mehmood\OneDrive\Documents\Lifelytix\data'

# === Load CSV files ===
demo = pd.read_csv(os.path.join(data_path, 'DEMO_I.csv'))
paq = pd.read_csv(os.path.join(data_path, 'PAQ_I.csv'))
diet = pd.read_csv(os.path.join(data_path, 'DR1IFF_I.csv'))
bp = pd.read_csv(os.path.join(data_path, 'BPX_I.csv'))
mcq = pd.read_csv(os.path.join(data_path, 'MCQ_I.csv'))
bmx = pd.read_csv(os.path.join(data_path, 'BMX_I.csv'))

# === Merge datasets on 'SEQN' participant ID ===
merged_df = demo.merge(paq, on='SEQN', how='left')\
                .merge(diet, on='SEQN', how='left')\
                .merge(bp, on='SEQN', how='left')\
                .merge(mcq, on='SEQN', how='left')\
                .merge(bmx, on='SEQN', how='left')

# === Save merged version (optional) ===
merged_path = os.path.join(data_path, 'merged_lifestyle_data.csv')
merged_df.to_csv(merged_path, index=False)
print(f"✅ Merged dataset saved to: {merged_path}")

# === Drop columns with >30% missing values ===
threshold = 0.70 * merged_df.shape[0]
clean_df = merged_df.dropna(axis=1, thresh=threshold)

# === Fill missing values ===
for col in clean_df.select_dtypes(include=['float64', 'int64']):
    clean_df[col].fillna(clean_df[col].median(), inplace=True)

for col in clean_df.select_dtypes(include=['object']):
    clean_df[col].fillna(clean_df[col].mode()[0], inplace=True)

# === Remove duplicates ===
clean_df.drop_duplicates(subset='SEQN', inplace=True)

# === Calculate BMI using correct metric columns ===
if {'BMXWT', 'BMXHT'}.issubset(clean_df.columns):
    clean_df['BMI'] = clean_df['BMXWT'] / (clean_df['BMXHT'] / 100) ** 2
    print("✅ BMI calculated using BMXWT and BMXHT")
else:
    print("❌ Missing BMXWT or BMXHT. BMI not calculated.")

# === Save final cleaned data ===
final_path = os.path.join(data_path, 'cleaned_lifestyle_data_with_BMI.csv')
clean_df.to_csv(final_path, index=False)
print(f"✅ Final cleaned dataset saved to: {final_path}")

# === Show data summary ===
print("\n📋 Final Columns:")
print(clean_df.columns.tolist())
print("\n🔍 Sample Data:")
print(clean_df.head())
