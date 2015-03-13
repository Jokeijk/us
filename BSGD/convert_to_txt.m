% this script visualizes U and V
[U, V, ~, ~, ~] = READ_UV();
% movie_all = read_movie('movies.txt');
[movieID,movieName,movieGenre,Genres] = read_movie('movies.txt');
%%
% now find the 500 most popular movies
popular = find_popular(movieName, 100, false);
%%
% now project U, V into 2-dim spaces
[A, ~, ~] = svd(V);
V_tilde = transpose(A(:, 1:2)) * V;
U_tilde = transpose(A(:, 1:2)) * U;
%%
% read the movies that linghu has watched
fid = fopen('linghu.txt', 'r');
linghu = fscanf(fid, '%d');
fclose(fid);
%%
% now, visualize them
V_popular = V(:, popular);
scatter(V_popular(1,:), V_popular(2,:))
%%
% Python seems to have better plotting tools, switching to Python...
dlmwrite('U.txt', U, '\t')
dlmwrite('V.txt', V, '\t')
