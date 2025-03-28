import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, List

def load_financial_phrasebank(file_path: str = "data/dataset/all-data.csv") -> Tuple[List[str], List[str], List[int], List[int]]:
    """
    Load and preprocess the Financial PhraseBank dataset.

    Args:
        file_path (str): Path to the CSV file containing the dataset.

    Returns:
        Tuple containing:
        - train_texts: List of training sentences
        - test_texts: List of test sentences
        - train_labels: List of training labels
        - test_labels: List of test labels
    """
    try:
        df = pd.read_csv(file_path)
        
        if 'sentence' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV file must contain 'sentence' and 'label' columns")
        
        label_map = {"positive": 2, "neutral": 1, "negative": 0}
        df['label'] = df['label'].map(label_map)
        
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            df['sentence'].tolist(),
            df['label'].tolist(),
            test_size=0.2,
            random_state=42,
            stratify=df['label']
        )
        
        return train_texts, test_texts, train_labels, test_labels
    
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {file_path} was not found. Please check the path and try again.")
    except Exception as e:
        raise Exception(f"An error occurred while loading the dataset: {str(e)}")

# Example usage
if __name__ == "__main__":
    try:
        train_texts, test_texts, train_labels, test_labels = load_financial_phrasebank()
        print(f"Loaded {len(train_texts)} training samples and {len(test_texts)} test samples.")
    except Exception as e:
        print(f"Error: {str(e)}")
