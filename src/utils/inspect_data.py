import pandas as pd

def inspect_dataset():

    print("Loading dataset...")
    df = pd.read_csv('src/data/ai4i2020.csv')
    
    print("\nDataset Shape:", df.shape)
    print("\nColumns in dataset:")
    for col in df.columns:
        print(f"- {col}")
        
    print("\nFirst few rows of data:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    inspect_dataset()