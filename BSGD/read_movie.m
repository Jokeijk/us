% Read movie.txt data file
function [movieID,movieName,movieGenre,Genres] = read_movie(filename)
if nargin < 1
    filename = 'movies.txt';
end
fid = fopen(filename);
m = textscan(fid,['%d %s ',repmat('%d',1,19)],'delimiter','\t');
fclose(fid);

movieID   = m{1};
movieName = m{2};
movieGenre= [m{3:end}];
Genres    = {'unknown','Action','Adventure','Animation','Children''s','Comedy',...
    'Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical',...
    'Mystery','Romance','Sci-Fi','Thriller','War','Western'};
