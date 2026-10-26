import os
import time
import requests
import pandas as pd
import numpy as np

class LLMAgent:
    def __init__(self, api_key=None, model="gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1/chat/completions")
        
        if "chat/completions" in self.url:
            self.embedding_url = self.url.replace("chat/completions", "embeddings")
        else:
            self.embedding_url = "https://api.openai.com/v1/embeddings"

        self.model = model

    def _request(self, prompt, retries=3):
        headers = {
            'Authorization': 'Bearer ' + self.api_key,
            'Content-Type': 'application/json',
        }
        data = {
            'model': self.model,
            'messages': [{"role": "user", "content": prompt}],
        }
        for i in range(retries):
            try:
                response = requests.post(self.url, headers=headers, json=data, timeout=60)
                response.raise_for_status()
                message = response.json()
                content = message['choices'][0]['message']['content']
                return content
            except Exception as e:
                print(f"Request failed, retrying {i+1}/{retries} times: {e}")
                time.sleep(2)
        return None

    def generate_latent_profile(self, history_items, item_attributes):
        history_string = "User history:\n"
        for item_id in history_items:
            row = item_attributes.loc[item_attributes['id'] == item_id]
            if not row.empty:
                year = row['year'].values[0]
                title = row['title'].values[0]
                history_string += f"[{item_id}] {year}, {title}\n"

        output_format = (
            f"Please infer the user's information in the format below. Base your answers on the movie history and do not use 'unknown' as a response. "
            f"You must infer from the patterns and information given.\n"
            f"Final output format:\n"
            f"{{\n"
            f"  'summary': interaction history summary, \n"
            f"  'age': user's likely age, \n"
            f"  'gender': user's likely gender, \n"
            f"  'liked genre': user's likely liked genre, \n"
            f"  'disliked genre': user's likely disliked genre, \n"
            f"  'liked directors': user's likely liked directors, \n"
            f"  'country': user's likely country, \n"
            f"  'language': user's likely language, \n"
            f"  'explanation': why you made such guesses based on the history\n"
            f"}}\n"
        )

        prompt = "You are a Latent Profile Generator (LPG) in MIRAGE. Generate a coarse-grained global profile from the interaction history below.\n"
        prompt += history_string
        prompt += output_format

        return self._request(prompt)

    def summarize_candidate_pool(self, retrieved_item_ids, item_attributes):
        subset = item_attributes[item_attributes['id'].isin(retrieved_item_ids)]

        if 'genre' in subset.columns:
            all_genres = subset['genre'].dropna().astype(str).tolist()
            flat_genres = [g.strip() for sublist in all_genres for g in sublist.replace('|', ',').split(',')]
            genre_counts = pd.Series(flat_genres).value_counts().head(20)
            genres_str = ", ".join(genre_counts.index.tolist())
        else:
            genres_str = "Information not available"

        if 'category' in subset.columns:
            all_cats = subset['category'].dropna().astype(str).tolist()
            flat_cats = [c.strip() for sublist in all_cats for c in sublist.replace('|', ',').split(',')]
            cat_counts = pd.Series(flat_cats).value_counts().head(20)
            cats_str = ", ".join(cat_counts.index.tolist())
        else:
            cats_str = "Information not available"

        if 'year' in subset.columns:
            years = pd.to_numeric(subset['year'], errors='coerce').dropna()
            if not years.empty:
                min_year = int(years.min())
                max_year = int(years.max())
                buckets = range((min_year // 10) * 10, (max_year // 10) * 10 + 20, 10)
                year_counts = years.groupby(pd.cut(years, buckets)).count()
                valid_buckets = year_counts[year_counts > 0]
                years_str = ", ".join([f"{int(interval.left)}-{int(interval.right)}" for interval in valid_buckets.index])
            else:
                years_str = "Information not available"
        else:
            years_str = "Information not available"

        summary_text = (
            "Module: Global Profile Modeling Summary\n\n"
            f"Candidate Genres: The candidate items belong to the following genres: [{genres_str}]\n"
            f"Candidate Categories: The items can be categorized into: [{cats_str}]\n"
            f"Years of release: The movies were released in the following periods: [{years_str}]"
        )
        return summary_text

    def infer_multi_view_intent(self, history_items, item_attributes, candidate_summary):
        history_string = ""
        recent_items = history_items[-10:]
        for i, item_id in enumerate(recent_items):
            row = item_attributes.loc[item_attributes['id'] == item_id]
            if not row.empty:
                year = row['year'].values[0]
                title = row['title'].values[0]
                genre = row['genre'].values[0] if 'genre' in row else "Unknown"
                history_string += f"No.{i+1}: Title: {title}; Year: {year}; Genre: {genre}\n"

        prompt = (
            "Module: Multi-View Intent Reasoner (MVIR)\n\n"
            "Instruction: Please infer the user's watching intent based on user history and candidate information. "
            "You must analyze the user's intent from three orthogonal dimensions: \n"
            "1. Similarity Intent (Exploitation): Focus on the user's core interests and comfort zone.\n"
            "2. Complementarity Intent (Extension): Focus on related but different content based on semantic logic.\n"
            "3. Diversity Intent (Exploration): Focus on high-quality content in fields the user has rarely touched.\n\n"
            "Begin by summarizing the relevant information, then provide your intent reasoning predictions using the user history details, "
            "candidate sets, and your general knowledge.\n\n"
            f"User history:\n{history_string}\n"
            f"Candidate summary:\n{candidate_summary}\n"
            "Output Format: JSON with keys 'Similarity_Intent', 'Complementarity_Intent', 'Diversity_Intent'. Ensure the output is valid JSON.\n"
        )
        return self._request(prompt)

    def encode(self, text):
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            headers = {
                'Authorization': 'Bearer ' + self.api_key,
                'Content-Type': 'application/json',
            }
            data = {
                'input': batch,
                'model': 'text-embedding-3-large'
            }
            
            for attempt in range(3):
                try:
                    response = requests.post(self.embedding_url, headers=headers, json=data, timeout=60)
                    response.raise_for_status()
                    result = response.json()
                    batch_embeddings = [item['embedding'] for item in sorted(result['data'], key=lambda x: x['index'])]
                    embeddings.extend(batch_embeddings)
                    break
                except Exception as e:
                    print(f"Embedding batch {i} failed, attempt {attempt+1}: {e}")
                    time.sleep(2)
            else:
                raise RuntimeError("Failed to generate embeddings")

        if isinstance(text, str):
            return np.array(embeddings[0])
        else:
            return np.array(embeddings)
