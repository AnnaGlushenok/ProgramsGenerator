import re
import hdbscan
import pdfplumber
import umap
import Config
import json
import numpy as np

from openai import OpenAI
from Config import Config
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist

client = OpenAI(base_url=Config.URL, api_key=Config.OPEN_ROUTER_API_KEY)


# TODO embeding api
def process_cash(path, max_sentences=5):
    print("read chunks")
    chunks = np.load("chunks.npy")
    print("read embeddings")
    embeddings = np.load("embeddings.npy")
    labels = clusterize(embeddings)
    answer(embeddings, chunks, labels)
    print("Done!")


def process(paths, max_sentences=5):
    print("Пошло-поехало")
    texts = ",".join([read(path) for path in paths])
    chunks = get_chunks(texts, max_sentences)
    np.save("chunks.npy", chunks)
    embeddings = get_embeddings(chunks)
    np.save("embeddings.npy", embeddings)
    labels = clusterize(embeddings)
    answer(embeddings, chunks, labels)
    print("Готово!")


def read(path):
    print(f"read file {path}")
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text


def get_chunks(text, max_sentences=5):
    print("create chunks")
    sentences = re.split(Config.PATTERN_SENTENCE, text)
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunks.append(" ".join(sentences[i:i + max_sentences]))
    return chunks


def get_embeddings(chunks):
    print("create embeddings")
    model = SentenceTransformer(Config.TRANSFORMER_MODEL)
    embeddings = model.encode(chunks)
    return embeddings


# TODO найти модель, протестить!
def get_embeddings_api(chunks):
    embedding = client.embeddings.create(
        model="sentence-transformers/paraphrase-minilm-l6-v2",
        input=chunks,
        encoding_format="float"
    )
    return embedding.data[0].embedding


def clusterize(embeddings):
    print("Кластеризуется")
    umap_model = umap.UMAP(n_neighbors=20, n_components=15, metric='cosine')
    emb_small = umap_model.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=4, metric='euclidean')
    labels = clusterer.fit_predict(emb_small)

    print("Кол-во кластеров", len(set(labels)) - 1)
    return labels


def get_central_chunks(embeddings, labels, chunk_texts, n_central=5):
    print("Ищу центроиды кластеров")
    cluster_embeddings = defaultdict(list)
    cluster_indices = defaultdict(list)

    for i, (emb, label) in enumerate(zip(embeddings, labels)):
        if label != -1:
            cluster_embeddings[label].append(emb)
            cluster_indices[label].append(i)

    central_chunks = {}
    for label, emb_list in cluster_embeddings.items():
        emb_array = np.array(emb_list)
        centroid = emb_array.mean(axis=0).reshape(1, -1)

        distances = cdist(emb_array, centroid, metric='cosine').flatten()
        closest_idx = np.argsort(distances)[:n_central]

        texts = [chunk_texts[cluster_indices[label][i]] for i in closest_idx]
        central_chunks[label] = texts

    return central_chunks

# TODO уточнить возвращаемую структуру списка
def sort(topics, responses):
    print("СортИрует")
    prompt = f""" 
        Упорядочи темы учебной программы логически: от базовых к продвинутым.
        Темы:
        {topics}
        Верни отсортированный список.
        """
    topics = ask_LLM(prompt)
    return sorted(responses, key=lambda x: topics.index(x['topic']))




def ask_LLM(prompt):
    return client.chat.completions.create(
        model=Config.LLM_MODELS["sherlock"],
        temperature=0.2,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    ).choices[0].message.content


def answer(embeddings, chunks, labels):
    central_chunks = get_central_chunks(embeddings, labels, chunks, n_central=5)
    topics = []
    responses = []

    cluster_keys = list(central_chunks.keys())
    i = 0
    while i < len(cluster_keys):
        # if i == 3: break
        cluster_id = cluster_keys[i]
        chunk_list = central_chunks[cluster_id]
        print(f"Итерация №{i}")
        prompt = f"""
               На основе следующих текстов сформируй учебную тему и сделай это чистым json объектом.
               Выдели:
               1. Название темы (короткое) topic
               2. Краткое описание темы description
               3. Подтемы или шаги изучения (список) subtopic

               Тексты:
               {". ".join(chunk_list)}
               """
        answer = ask_LLM(prompt)
        try:
            resp = json.loads(answer)
            responses.append(resp)
            topics.append(resp["topic"])
            i += 1
        except ValueError:
            print(answer)
            print(f"Ошибка JSON, повторим для кластера {cluster_id}...")

    info = sort(topics, responses)
    with open('example-finale.json', "a", encoding='utf-8') as f:
        f.write("[")
        f.write(json.dumps(info, ensure_ascii=False, indent=2))
        f.write("]")


process_cash(["src/BD.pdf", "src/MySQL_по_максимуму_2023.pdf"])
# process(["src/BD.pdf", "src/MySQL_по_максимуму_2023.pdf"])
# process("src/BD.pdf") # Для первого запуска
