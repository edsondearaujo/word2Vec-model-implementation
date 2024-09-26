import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Carregar o dataset
b2wCorpus = pd.read_csv("data/b2w-10k.csv")

# Selecionar as colunas relevantes
b2wCorpus = b2wCorpus[['review_text', 'recommend_to_a_friend']]

# Remover NaN e converter recomendação para binário
b2wCorpus = b2wCorpus.dropna()
b2wCorpus['recommend_to_a_friend'] = b2wCorpus['recommend_to_a_friend'].apply(lambda x: 1 if x == 'Yes' else 0)

# Dividir os dados em treino e teste (75% treino, 25% teste)
train_data, test_data = train_test_split(b2wCorpus, test_size=0.25, random_state=42)

# Separar as colunas de features (texto) e labels (recomendação)
train_texts = train_data['review_text'].values
train_labels = train_data['recommend_to_a_friend'].values
test_texts = test_data['review_text'].values
test_labels = test_data['recommend_to_a_friend'].values
