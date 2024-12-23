from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
import os

pd.set_option('display.max_columns', None)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
df = pd.read_csv('eth_corpus.csv')
documents = df['text'].astype(str)

vectorizer = CountVectorizer(min_df=2, stop_words='english')
X = vectorizer.fit_transform(documents)

lda = LatentDirichletAllocation(n_components=8, random_state=42)
lda.fit(X)

def print_topics(model, vectorizer, top_n=6):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names_out()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])

print_topics(lda, vectorizer)

print("=====================")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(documents)
umap_model = UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=4, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

topic_model = BERTopic(
  embedding_model=embedding_model,
  umap_model=umap_model,
  hdbscan_model=hdbscan_model,
  vectorizer_model=vectorizer,
  top_n_words=7,
  verbose=True
)

topics, probs = topic_model.fit_transform(documents, embeddings)

print(topic_model.get_topic_info())
for i in range (10):
    print(topic_model.get_topic(topic=i))

fig = topic_model.visualize_topics()
fig.write_image('topic_modelling_eth_corpus.png')