% Read the data file
function [userID,movieID,rate] = read_data(filename)
if nargin < 1
    filename = 'data.txt';
end

fid = fopen(filename);
m = textscan(fid,'%f %f %f');
fclose(fid);

userID = m{1};
movieID= m{2};
rate = m{3};