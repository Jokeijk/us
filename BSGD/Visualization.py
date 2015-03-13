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

# now find the movies we've watched
linghu = pd.read_table('linghu.txt', header=None)
dunzhu = pd.read_table('dunzhu.txt', header=None)
yiran = pd.read_table('yiran.txt', header=None)

# now visualize the most popular movies
N = 50
popular_index = set(find_popular(movie_all, N))
seen = np.array(list(
    set(linghu[0]-1).union(set(dunzhu[0]-1)).union(set(yiran[0]-1)).intersection(popular_index)
))
movie_seen = movie_all[['title']].iloc[seen].as_matrix()
plt.scatter(V_tilde[0, seen], V_tilde[1, seen], marker='+', s=50)
offset = 10
for label, x, y in zip(movie_seen,
                       V_tilde[0, seen].transpose(),
                       V_tilde[1, seen].transpose()):
    print label
    plt.annotate(
        label[0][:-7],
        xy=(x, y), xytext=(-offset, offset),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        # arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
    )
plt.axhline()
plt.axvline()
plt.show()