% User-Movie matrix
function [M,N,Y_s] = sparse_matrix(userID_all, movieID_all, userID1, movieID1,rate1)
M = length(userID_all);
N = length(movieID_all);

% the indices of moveID1 in movieID_all
[a, sub_2] = ismember(movieID1,movieID_all);
if isempty(find(a,1))
    error('Not all the movie found!');
end

% the indices of useID1 in userID_all
[a, sub_1] = ismember(userID1,userID_all);
if isempty(find(a,1))
    error('Not all the user found!');
end

Y_s     = [sub_1(:) sub_2(:) rate1(:)]; 


end