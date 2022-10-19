import pandas as pd
import numpy as np
dataframe1 = pd.read_csv('tmdb_5000_credits.csv')
dataframe2 = pd.read_csv('tmdb_5000_movies.csv')
dataframe1.columns=['movie_id','title','cast','crew']
dataframe2 = dataframe2.merge(dataframe1,on='movie_id')

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
dataframe2['overview'] = dataframe2['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(dataframe2['overview'])

from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(dataframe2.index, index=dataframe2['title_x']).drop_duplicates()

def recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:11]
    movie_indices = [i[0] for i in similarity_scores]
    return dataframe2['title_x'].iloc[movie_indices]