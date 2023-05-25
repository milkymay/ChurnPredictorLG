import json
import os

import joblib
import numpy as np
import openai
import pandas as pd
from sklearn.decomposition import PCA

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def get_embedding(text, model="text-embedding-ada-002"):
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']


def generate_embeddings(texts):
    embeddings = []
    for text in texts:
        text = text.replace("\n", " ")
        embeddings.append(get_embedding(text))
    return np.array(embeddings)


def reduce_dimensions(embeddings, n_components=20):
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings, pca


class ChattyEmbeddingManager:
    def __init__(self, conf_path, raw_lessons, log_path):
        self.raw_lessons = raw_lessons
        meaningful = raw_lessons[raw_lessons['comment_for_parent'] != ""]
        comments = meaningful['comment_for_parent']
        self.indices = pd.DataFrame({'index': raw_lessons.index}).reset_index(drop=True)
        self.parsed_result = None
        self.comments = comments
        self.log_path = log_path
        data = json.load(open(conf_path, 'r'))
        openai.api_key = data['OPENAI_API_KEY']
        self.time = 0
        self.result = []

    def add_embeddings(self):
        model_name = "text-embedding-ada-002"
        batch_size = 1000
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)
        df = pd.DataFrame(self.comments)
        n_batches = (df.shape[0] // batch_size) + 1
        batches = np.array_split(df["comment_for_parent"].values, n_batches)
        for i in range(1, len(batches)):
            batch = batches[i]
            print(f"Processing batch {i + 1}/{n_batches}")
            embeddings = generate_embeddings(batch)
            np.savetxt(self.log_path + f"/embeddings_{i + 1}.csv", embeddings,
                       delimiter=",")
            reduced_embeddings, pca = reduce_dimensions(embeddings, n_components=20)
            np.savetxt(self.log_path + f"/reduced_embeddings_{i + 1}.csv",
                       reduced_embeddings, delimiter=",")
            pca_filename = self.log_path + f"/pca_model_{i + 1}.pkl"
            joblib.dump(pca, pca_filename)
            del embeddings, reduced_embeddings, pca
            reduced_embeddings = []
            for j in range(n_batches):
                reduced_embeddings.append(
                    np.loadtxt(self.log_path + f"/reduced_embeddings_{j + 1}.csv",
                               delimiter=","))
            reduced_embeddings = np.vstack(reduced_embeddings)
            pca_models = []
            for j in range(n_batches):
                pca_models.append(joblib.load(self.log_path + f"/pca_model_{j + 1}.pkl"))
            pca_model = PCA(n_components=20)
            pca_model.fit(np.vstack([model.explained_variance_ for model in pca_models]))

            final_reduced_embeddings = pca_model.transform(reduced_embeddings)

            df = pd.DataFrame(final_reduced_embeddings,
                              columns=[self.log_path + f"/embedding_{i}" for i
                                       in
                                       range(20)])
            df.to_csv(self.log_path + '/reduced_embeddings.csv', index=False)
            return self.parse_result()

    def parse_result(self):
        embeddings_df = pd.read_csv(self.log_path + '/reduced_embeddings.csv')

        result_df = self.indices.merge(embeddings_df, left_index=True, right_index=True)
        result_df.columns = result_df.columns.str.replace(self.log_path + '/',
                                                          '')
        merged_df = pd.merge(self.indices, result_df, on='index', how='outer')
        merged_df = merged_df.fillna(0)
        merged_df_reind = merged_df.set_index(merged_df['index'])
        merged_df_reind = merged_df_reind.drop(columns=['index'])
        answer = self.raw_lessons.merge(merged_df_reind, left_index=True, right_index=True)
        return answer
