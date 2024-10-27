# dataset.py

from datasets import load_dataset

def load_clerc_dataset(limit=1250):
    dataset = load_dataset("jhu-clsp/CLERC", split='train')
    dataset = dataset.shuffle(seed=42).select(range(min(limit, len(dataset))))
    limited_data = {
        'query': dataset['query'],
        'positive_passages': [passage['text'] for passages in dataset['positive_passages'] for passage in passages if passages],
        'negative_passages': [passage['text'] for passages in dataset['negative_passages'] for passage in passages if passages],
    }
    return limited_data
