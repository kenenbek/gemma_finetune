from data import KyrgyzDataLoader
import string
import jiwer
from tqdm import tqdm

if __name__ == '__main__':
    loader = KyrgyzDataLoader()
    splits = loader.load_from_hf_dataset('../misspelled_kg_dataset/', num_samples=None)

    print(splits)