import re
import pdfplumber
import hdbscan
import umap
import requests
import json
import os
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from openai import OpenAI

PATTERN_SENTENCE = r'[.!?]\s+'
TRANSFORMER_MODEL = "intfloat/multilingual-e5-base"
# base_url="https://api.poe.com/v1"

load_dotenv()
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
print(OPEN_ROUTER_API_KEY)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPEN_ROUTER_API_KEY,
)

models = {
    "GPT": "openai/gpt-4o",  #
    "Deepseek": "deepseek/deepseek-chat-v3.1:free",  #
    "Gemini": "google/gemini-2.0-flash-exp:free",  #
    "Polaris": "openrouter/polaris-alpha"
}


def process(path, max_sentences=5):
    # print("read file")
    # text = read(path)
    # v = "Довольно скоро на смену иерархической и сетевой моделям пришла принципиально новая модель данных – реляционная. Реляционная модель данных – логическая модель, основанная на множестве взаимосвязанных именованных отношений. В свою очередь, реляционная база данных – база данных, основанная на реляционной модели данных. Термин «реляционный» означает, что теория основана на математическом понятии «отношение». Отношение – это информационная модель реального объекта предметной области, формально представленная множеством однотипных кортежей. Кортеж отношения представляет собой экземпляр моделируемого объекта, свойства которого определяются значениями соответствующих атрибутов («полей») кортежа. Связи между кортежами отношений (при их наличии) реализуются через простой механизм «внешних ключей», являющихся, по существу, ссылками на атрибуты связываемых кортежей отношений. Переменная отношения находится в первой нормальной форме (1НФ) тогда и только тогда, когда в любом допустимом значении этой переменной каждый кортеж отношения содержит только одно значение для каждого из атрибутов. Согласно определению отношений, любое отношение автоматически уже находится в 1НФ согласно свойствам отношений: в отношении нет одинаковых кортежей; кортежи не упорядочены; атрибуты не упорядочены и различаются по наименованию; все значения атрибутов атомарны. Данные свойства и являются свойствами первой нормальной формы. Если упростить определение, таблица приведена к первой нормальной форме, если в ней отсутствуют повторяющиеся группы, и все значения, хранимые в ней, неделимы (атомарны).Вернемся к примеру. В ходе логического моделирования на первом шаге предложено хранить данные в одном сводном отношении, имеющем следующие атрибуты: Инструкция SELECT, предназначенная для выборки данных из одной или нескольких таблиц или представлений, является основным элементом языка SQL. В простейшем случае результат запроса может полностью повторять содержимое одной таблицы, может представлять собой объединение нескольких таблиц, определенную выборку из исходных данных и т.д. Помимо этого, запросы SELECT активно используются при создании ряда ключевых объектов базы данных (представлений, хранимых процедур и триггеров). Базовая синтаксическая конструкция запроса выглядит следующим образом: SELECT [DISTINCT|ALL] { имя_столбца [AS псевдоним] [,…] |функция_агрегирования [AS псевдоним] [,…] |выражение_для_вычисления_значения [AS псевдоним] [,…] |спецификатор.* } FROM {имя_таблицы [AS псевдоним] [,…] |имя_представления [AS псевдоним][,…]} [WHERE условия_отбора] [GROUP BY имя_столбца [,…]] [HAVING условие] [ORDER BY имя_столбца [ASC | DESC] [,…]]"
    print("create chunks")
    # chunks = get_chunks(text, max_sentences)
    # np.save("chunks.npy", chunks)
    chunks = np.load("chunks.npy")
    print("create embeddings")
    # embeddings = get_embeddings(chunks)
    embeddings = np.load("embeddings.npy")
    # np.save("embeddings.npy", embeddings)
    labels = cluster(embeddings)
    answer(chunks, labels)
    print("Done!")


def read(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text


def get_chunks(text, max_sentences=5):
    sentences = re.split(PATTERN_SENTENCE, text)
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunks.append(" ".join(sentences[i:i + max_sentences]))
    return chunks


def get_embeddings(chunks):
    model = SentenceTransformer(TRANSFORMER_MODEL)
    embeddings = model.encode(chunks)
    return embeddings


def cluster(embeddings):
    umap_model = umap.UMAP(n_neighbors=15, n_components=15, metric='cosine')
    emb_small = umap_model.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=4, metric='euclidean')
    labels = clusterer.fit_predict(emb_small)

    print("Кол-во кластеров", len(set(labels)) - 1)
    return labels


def answer(chunks, labels):
    cluster_texts = defaultdict(list)
    for text, label in zip(chunks, labels):
        if label != -1:
            cluster_texts[label].append(text)

    # for cluster_id, chunk_list in cluster_texts.items():
    #     print(f"Кластер {cluster_id}:")
    #     for chunk in chunk_list[:3]:
    #         print(" -", chunk)
    i = 0
    for cluster_id, chunk_list in cluster_texts.items():
        if i == 3:
            break
        i += 1
        print(f"\n********************** {i} **********************")
        prompt = f"""
        На основе следующих текстов сформируй учебную тему.
        Выдели:
        1. Название темы (короткое)
        2. Краткое описание темы
        3. Подтемы или шаги изучения (список)
 
        Тексты:
        {chunk_list[:5]} 
        """

        response = client.chat.completions.create(
            model=models["Polaris"],
            # max_tokens=1000,
            temperature=0.2,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        print(response.choices[0].message.content)


process("src/BD1.pdf")
