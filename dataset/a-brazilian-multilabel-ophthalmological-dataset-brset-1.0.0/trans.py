import pandas as pd

# Define the age categorization functions
def _age_to_4class(age):
    if age <= 47:
        return 0
    elif age <= 61:
        return 1
    elif age <= 71:
        return 2
    else:
        return 3

def _age_to_3class(age):
    if age <= 52:
        return 0
    elif age <= 67:
        return 1
    else:
        return 2

def _age_to_2class(age):
    if age <= 61:
        return 0
    else:
        return 1

# Function to process the CSV file
def process_csv(input_csv, output_csv):
    # Read the CSV file
    df = pd.read_csv(input_csv)

    # Check if 'patient_sex' column exists
    if 'patient_sex' not in df.columns:
        raise ValueError("CSV does not have a 'patient_sex' column")

    # Apply the age categorization functions
    df['AgeClass4'] = df['patient_age'].apply(_age_to_4class)
    df['AgeClass3'] = df['patient_age'].apply(_age_to_3class)
    df['AgeClass2'] = df['patient_age'].apply(_age_to_2class)
    print(df['AgeClass3'].value_counts())
    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)

# Example usage
process_csv('train.csv', 'train.csv')
process_csv('val.csv', 'val.csv')
