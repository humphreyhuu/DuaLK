import requests
import json
import os
import time


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
with open(os.path.join(project_root, "resources", "openai.key"), 'r') as f:
    key = f.readlines()[0]  # [:-1]


def embedding_retriever(term, retries=3, delay=5):
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }
    payload = {
        "input": term,
        "model": "text-embedding-3-small"
    }

    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)

            if response.status_code == 200:
                embedding = response.json()["data"][0]['embedding']
                num_tokens = response.json()["usage"]["prompt_tokens"]
                return embedding, num_tokens
            else:
                print(f"Failed to retrieve embeddings: {response.status_code}")
                print(response.text)
                time.sleep(delay)  # Wait before retrying
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            time.sleep(delay)  # Wait before retrying

    print("Failed to retrieve embeddings after retries.")
    return None, None
