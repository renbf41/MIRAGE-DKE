import os
import ast
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

INTENT_KEYS = [('Similarity_Intent', 'Intent_Text_Sim'), ('Complementarity_Intent', 'Intent_Text_Comp'),
               ('Diversity_Intent', 'Intent_Text_Div')]


def embed(texts, model, url, batch=64):
    out = []
    for i in tqdm(range(0, len(texts), batch), desc=f'Encoding with {model}'):
        response = requests.post(url, json={'model': model, 'input': texts[i:i + batch]}, timeout=600)
        response.raise_for_status()
        out.extend(response.json()['embeddings'])
    emb = np.asarray(out, dtype=np.float32)
    return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)


def intent_texts(entry):
    if isinstance(entry, str):
        try:
            entry = json.loads(entry)
        except ValueError:
            try:
                entry = ast.literal_eval(entry)
            except (ValueError, SyntaxError):
                return [entry or 'unknown'] * 3
    return [str(entry.get(a, entry.get(b, ''))) or 'unknown' for a, b in INTENT_KEYS]


def main():
    parser = argparse.ArgumentParser(description='Encode item titles and intent texts in a shared space')
    parser.add_argument('--data_dir', type=str, default=os.environ.get('DATA_DIR', '.'), help='Data directory')
    parser.add_argument('--dataset', type=str, default='netflix_data', help='Dataset name')
    parser.add_argument('--model', type=str, default='bge-m3', help='Embedding model served by Ollama')
    parser.add_argument('--url', type=str, default='http://localhost:11434/api/embed', help='Embedding endpoint')
    args = parser.parse_args()

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    n_users, n_items = pickle.load(open(os.path.join(dataset_dir, 'train_mat'), 'rb')).shape
    attributes = pd.read_csv(os.path.join(dataset_dir, 'netflix_image_text', 'item_attribute.csv'),
                             header=None, names=['id', 'year', 'title'])
    titles = {int(r.id): f'{r.title} ({int(float(r.year))})' if pd.notna(r.year) else str(r.title)
              for r in attributes.itertuples()}
    item_texts = [titles.get(i, 'unknown movie') for i in range(n_items)]

    intents = pickle.load(open(os.path.join(dataset_dir, 'mvir_multi_view_intent_dict'), 'rb'))
    texts = [t for u in range(n_users) for t in intent_texts(intents.get(u, ''))]

    item_emb = embed(item_texts, args.model, args.url)
    intent_emb = embed(texts, args.model, args.url).reshape(n_users, 3, -1)
    path = os.path.join(dataset_dir, 'text_embeddings.npz')
    np.savez_compressed(path, item=item_emb, intent=intent_emb)
    print(f'Saved {path}: items {item_emb.shape}, intents {intent_emb.shape}')


if __name__ == '__main__':
    main()
