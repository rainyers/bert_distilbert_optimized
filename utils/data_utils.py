
from datasets import load_dataset

def load_ag_news():
    dataset = load_dataset("ag_news")
    return dataset
