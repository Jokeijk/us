function Graphchi
addpath('src');
[userID1,movieID1,rate1] = read_data('data.txt');
[movieID,movieName,movieGenre,Genres] = read_movie('movies.txt');
userID = unique(userID1);
[M,N,Y_s] = sparse_matrix(userID, movieID, userID1, movieID1,rate1);
write_matrix_MMF('a.in',M,N,Y_s);

!./DOIT.sh


end



