# this script visualize principal components of U and V

import re
import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from pylab import savefig

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
genre_list = ['Action', 'Adventure',
              'Animation', 'Children', 'Comedy', 'Crime', 'Documentary',
              'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
              'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western',
              'Unknown']
color_list = ['Red', 'Blue', 'Green', 'Yellow', 'Magenta', 'Gold',
              'SlateBlue', 'Purple', 'OrangeRed', 'Olive', 'Maroon', 'LightGreen',
              'Cyan', 'Wheat', 'Brown', 'Gray', 'DarkViolet', 'YellowGreen',
              'Black']
movie_all['genre'] = ''
for genre in genre_list:
    print genre
    # movie_all[movie_all[genre] == 1]['genre'] = genre
    movie_all.loc[movie_all[genre] == 1, 'genre'] = genre


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
N = 100
popular_index = set(find_popular(movie_all, N))

# now consider specific movie collections
yisong = pd.read_table('yisong.txt', header=None).as_matrix().reshape(-1) - 1
star_trek = np.array([221, 226, 227, 228, 229])
free_willy = np.array([34, 77])
god_father = np.array([126, 186])
batman = np.array([28, 230, 253, 402])
# terminator = np.array()
seen = np.array(list(
    set(list(linghu[0] - 1) +
        list(dunzhu[0] - 1) +
        list(yiran[0] - 1)).intersection(popular_index).union(
        set(list(star_trek) + list(free_willy) + list(god_father) + list(batman))
    )))
# seen = yisong
movie_seen = movie_all[['title', 'genre']].iloc[seen]
genre_sorted = movie_seen.groupby('genre').agg(
    'count').sort('title', ascending=False).reset_index()['genre']

font_size = 12
for i in xrange(len(genre_sorted)):
    genre_tmp = genre_sorted[i]
    movie_seen_tmp = movie_seen[movie_seen['genre'] == genre_tmp]
    seen_tmp = np.array(movie_seen_tmp.index)
    plt.scatter(V_tilde[0, seen_tmp], V_tilde[1, seen_tmp],
                marker='o', s=50, color=color_list[i], label=genre_tmp)
offset = 0
for label, x, y in zip(movie_seen[['title']].as_matrix(),
                       V_tilde[0, seen].transpose(),
                       V_tilde[1, seen].transpose()):
    # label_new = label[0][:-6].rstrip().rstrip(', The').rstrip(', A')
    label_new = re.sub(', A', '', re.sub(', The', '', label[0][:-6].rstrip()))
    print label_new
    plt.annotate(
        label_new,
        xy=(x, y), xytext=(-offset, offset),
        size=font_size,
        textcoords='offset points', ha='right', va='bottom',
        # bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.5),
        # arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
    )
plt.axhline()
plt.axvline()
plt.legend(loc=1, scatterpoints=1, prop={'size': font_size})
plt.show()

# now for each genre
for i in xrange(len(genre_list)):
    genre_tmp = genre_list[i]
    print genre_tmp
    movie_tmp = movie_all[movie_all['genre'] == genre_tmp]
    index_tmp = np.array(movie_tmp.index)
    fig = plt.figure()
    plt.scatter(V_tilde[0, index_tmp], V_tilde[1, index_tmp],
                marker='o', s=50, color=color_list[i], label=genre_tmp)
    plt.axis([-1, 1, -1, 1])
    plt.axhline()
    plt.axvline()
    fig.savefig(r'D:\Dropbox\CS 155\Projects\Git\us\BSGD\figures\%s.png' % genre_tmp)