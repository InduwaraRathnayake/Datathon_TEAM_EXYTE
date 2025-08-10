import pandas as pd

# Define all possible sub-categories
SUB_CATEGORIES = [
    'Potholes',
    'Damaged Signs',
    'Fallen Trees',
    'Graffiti',
    'Garbage',
    'Illegal Parking',
]

# Read the multi-label CSV
df = pd.read_csv('multi_label_dataset.csv')

# Create columns for each sub-category, fill with 0
for subcat in SUB_CATEGORIES:
    df[subcat] = 0

# Fill multi-hot encoding for each image
for idx, row in df.iterrows():
    subcats = [s.strip() for s in row['sub_category'].split(';')]
    for subcat in subcats:
        if subcat in SUB_CATEGORIES:
            df.at[idx, subcat] = 1

# Save encoded CSV
df.to_csv('multi_label_encoded.csv', index=False)

print('Multi-label encoding complete. Output: multi_label_encoded.csv')
