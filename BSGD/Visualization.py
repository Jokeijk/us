# this script visualize principal components of U and V

import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

data = pd.read_table('data.txt', header=None)
data.columns = ['user_id', 'movie_id', 'rating']

# now find the frequency for each movie
data_grouped = data[['user_id', 'movie_id']].groupby('movie_id').agg('count')
frequency = data_grouped.reset_index().rename(columns={'user_id': 'frequency'})
# data = data.merge(frequency)

movie_all = pd.read_table('movies.txt', header=None)
movie_all.columns = ['movie_id', 'title', 'Unknown', 'Action', 'Adventure',
                     'Animation', 'Children', 'Comedy', 'Crime', 'Documentary',
                     'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                     'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movie_all = movie_all.merge(frequency, on='movie_id')
genre_list = ['Unknown', 'Action', 'Adventure',
              'Animation', 'Children', 'Comedy', 'Crime', 'Documentary',
              'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
              'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movie_all['genre'] = ''
for genre in genre_list:
    print genre
    movie_all[movie_all[genre] == 1].loc['genre'] = 1


# find the N most popular movies
def find_popular(movie_all, N):
    movie_sorted = movie_all.sort('frequency', ascending=False)
    movie_popular = movie_sorted.iloc[:N]
    return (movie_popular[['movie_id']].as_matrix() - 1).reshape(-1)

# load U and V, project them to U_tilde, V_tilde
U = pd.read_table('U.txt', header=None).as_matrix()
V = pd.read_table('V.txt', header=None).as_matrix()

A, s, B = la.svd(V)
S = np.zeros(V.shape)
S[0:len(s), 0:len(s)] = np.diag(s)

V_tilde = A[:, :2].transpose().dot(V)
U_tilde = A[:, :2].transpose().dot(U)

# now visualize the most popular movies
N = 100
popular_index = find_popular(movie_all, N)
plt.scatter(V_tilde[0, popular_index], V_tilde[1, popular_index])
plt.axhline()
plt.axvline()
plt.show()