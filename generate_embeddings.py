import os
import pickle
import pandas as pd
import argparse
import torch
import numpy as np
import sys
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from llm_agent import LLMAgent
from dotenv import load_dotenv

load_dotenv()


def atomic_save(obj, path):
    temp_path = path + '.tmp'
    try:
        with open(temp_path, 'wb') as f:
            pickle.dump(obj, f)
        os.replace(temp_path, path)
    except Exception as e:
        print(f"Failed to save {path}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../MIRAGE-complete')
    parser.add_argument('--dataset', type=str, default='netflix_data')
    parser.add_argument('--limit', type=int, default=10, help='Limit number of users for demo')
    args = parser.parse_args()

    data_path = os.path.join(args.data_dir, args.dataset)

    print('Loading data...')
    train_mat = pickle.load(open(os.path.join(data_path, 'train_mat'), 'rb'))

    item_attr_path = os.path.join(data_path, 'netflix_image_text', 'item_attribute.csv')
    if not os.path.exists(item_attr_path):
        item_attr_path = os.path.join(data_path, 'item_attribute.csv')

    if os.path.exists(item_attr_path):
        try:
            item_attributes = pd.read_csv(item_attr_path)
            if 'id' not in item_attributes.columns:
                item_attributes = pd.read_csv(item_attr_path, names=['id', 'year', 'title'])
        except Exception as e:
            print(f"Error reading item_attribute.csv: {e}")
            sys.exit(1)
    else:
        print(f"Error: {item_attr_path} not found. Please provide the item attribute file.")
        sys.exit(1)

    agent = LLMAgent()

    print('\n--- Stage 1: Global Profile Modeling (LPG) ---')
    latent_profiles = {}
    latent_profile_save_path = os.path.join(data_path, 'lpg_latent_profile_dict')

    if os.path.exists(latent_profile_save_path):
        print(f"Loading existing latent profiles from {latent_profile_save_path}")
        latent_profiles = pickle.load(open(latent_profile_save_path, 'rb'))

    users_to_process = range(train_mat.shape[0])

    if len(latent_profiles) < len(users_to_process):
        print('Generating missing latent profiles...')

        def process_user(uid):
            _, history_items = train_mat[uid].nonzero()
            if len(history_items) == 0:
                return None
            try:
                profile_text = agent.generate_latent_profile(history_items, item_attributes)
                return uid, profile_text
            except Exception as e:
                print(f"Error for user {uid}: {e}")
                return None

        try:
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {
                    executor.submit(process_user, uid): uid
                    for uid in users_to_process if uid not in latent_profiles
                }

                completed_count = 0
                for future in tqdm(as_completed(futures), total=len(futures)):
                    result = future.result()
                    if result:
                        uid, profile_text = result
                        if profile_text:
                            latent_profiles[uid] = profile_text

                    completed_count += 1
                    if completed_count % 50 == 0:
                        atomic_save(latent_profiles, latent_profile_save_path)

            atomic_save(latent_profiles, latent_profile_save_path)
        except KeyboardInterrupt:
            print('\nInterrupted! Saving progress...')
            atomic_save(latent_profiles, latent_profile_save_path)
            sys.exit(0)

    print('Encoding latent profiles...')
    sorted_profiles = [latent_profiles.get(i, 'Unknown User') for i in range(train_mat.shape[0])]
    global_profile_embeddings = agent.encode(sorted_profiles)

    global_profile_emb_save_path = os.path.join(data_path, 'lpg_global_profile_embedding')
    pickle.dump(global_profile_embeddings, open(global_profile_emb_save_path, 'wb'))
    print(f"Saved Global Profile Embeddings to {global_profile_emb_save_path}")

    print('\n--- Stage 2: Multi-View Intent Reasoning (MVIR) ---')
    global_profile_embeddings_tensor = torch.tensor(global_profile_embeddings)

    multi_view_intents = {}
    multi_view_intent_save_path = os.path.join(data_path, 'mvir_multi_view_intent_dict')

    if os.path.exists(multi_view_intent_save_path):
        print(f"Loading existing intents from {multi_view_intent_save_path}")
        multi_view_intents = pickle.load(open(multi_view_intent_save_path, 'rb'))

    if len(multi_view_intents) < len(users_to_process):
        print('Generating missing multi-view intents...')

        def process_intent(uid):
            query_emb = global_profile_embeddings_tensor[uid].unsqueeze(0)
            sims = torch.nn.functional.cosine_similarity(query_emb, global_profile_embeddings_tensor)
            _, top_k_indices = torch.topk(sims, k=50)

            top_k_uids = top_k_indices.numpy()
            candidate_items = []
            for neighbor_id in top_k_uids:
                _, items = train_mat[neighbor_id].nonzero()
                candidate_items.extend(items)
            candidate_items = list(set(candidate_items))

            candidate_summary = agent.summarize_candidate_pool(candidate_items, item_attributes)

            _, history_items = train_mat[uid].nonzero()
            try:
                intent_response = agent.infer_multi_view_intent(history_items, item_attributes, candidate_summary)

                intent_dict = {}
                try:
                    intent_dict = json.loads(intent_response)
                except Exception:
                    match = re.search(r'```json\s*(\{.*?\})\s*```', intent_response, re.DOTALL)
                    if match:
                        try:
                            intent_dict = json.loads(match.group(1))
                        except Exception:
                            intent_dict = {}
                    else:
                        match = re.search(r'(\{.*\})', intent_response, re.DOTALL)
                        if match:
                            try:
                                intent_dict = json.loads(match.group(1))
                            except Exception:
                                intent_dict = {}

                required_keys = ['Similarity_Intent', 'Complementarity_Intent', 'Diversity_Intent']
                if not intent_dict or not any(k in intent_dict for k in required_keys):
                    intent_dict = {
                        'Similarity_Intent': intent_response,
                        'Complementarity_Intent': intent_response,
                        'Diversity_Intent': intent_response,
                    }
                else:
                    for k in required_keys:
                        if k not in intent_dict:
                            intent_dict[k] = 'General Intent'

                return uid, intent_dict
            except Exception as e:
                print(f"Error for user {uid}: {e}")
                return None

        try:
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {
                    executor.submit(process_intent, uid): uid
                    for uid in users_to_process if uid not in multi_view_intents
                }

                completed_count = 0
                for future in tqdm(as_completed(futures), total=len(futures)):
                    result = future.result()
                    if result:
                        uid, intent_text = result
                        if intent_text:
                            multi_view_intents[uid] = intent_text

                    completed_count += 1
                    if completed_count % 50 == 0:
                        atomic_save(multi_view_intents, multi_view_intent_save_path)

            atomic_save(multi_view_intents, multi_view_intent_save_path)
        except KeyboardInterrupt:
            print('\nInterrupted! Saving progress...')
            atomic_save(multi_view_intents, multi_view_intent_save_path)
            sys.exit(0)

    print('Encoding multi-view intents...')
    similarity_intents = []
    complementarity_intents = []
    diversity_intents = []

    for i in range(train_mat.shape[0]):
        intent_dict = multi_view_intents.get(i, {})
        if isinstance(intent_dict, str):
            similarity_intents.append(intent_dict)
            complementarity_intents.append(intent_dict)
            diversity_intents.append(intent_dict)
        else:
            similarity_intents.append(intent_dict.get('Similarity_Intent', 'General Intent'))
            complementarity_intents.append(intent_dict.get('Complementarity_Intent', 'General Intent'))
            diversity_intents.append(intent_dict.get('Diversity_Intent', 'General Intent'))

    print('Encoding Similarity Intents...')
    similarity_embeddings = agent.encode(similarity_intents)
    print('Encoding Complementarity Intents...')
    complementarity_embeddings = agent.encode(complementarity_intents)
    print('Encoding Diversity Intents...')
    diversity_embeddings = agent.encode(diversity_intents)

    multi_view_intent_embeddings = np.stack(
        [similarity_embeddings, complementarity_embeddings, diversity_embeddings],
        axis=1,
    )

    multi_view_intent_emb_save_path = os.path.join(data_path, 'mvir_multi_view_intent_embedding')
    pickle.dump(multi_view_intent_embeddings, open(multi_view_intent_emb_save_path, 'wb'))
    print(f"Saved Multi-View Intent Embeddings to {multi_view_intent_emb_save_path}")


if __name__ == '__main__':
    main()
