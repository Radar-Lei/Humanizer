from datasets import load_dataset
import pandas as pd

# Load the dataset
dataset = load_dataset("neuralwork/arxiver")

# Convert to pandas DataFrame for easier viewing
df = dataset['train'].to_pandas()

# Show the first 5 papers with title and abstract
print(df['markdown'].iloc[0])