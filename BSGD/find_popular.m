function popular = find_popular(movieName, N, write)
% first let's find the most popular movies, and add a column to movie_all
% to record them
[~, movieID_in_rating, ~] = read_data('data.txt');
movie_frequency = tabulate(movieID_in_rating);
[~, sorted_order] = sort(movie_frequency(:, 2), 'descend');
% movie_sort_by_frequency = sort(movie_all, 2)
popular = sort(sorted_order(1:N));
if write
    fid = fopen('popular.txt','w');
    for i = 1:N
        idx = popular(i);
        movie_id = movieID(idx);
        movie_name = movieName{idx};
        fprintf(fid,'%d ', movie_id);
        fprintf(fid, '%s', movie_name);
        fprintf(fid,'\n');
    end
    fclose(fid);
end