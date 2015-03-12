function [U,V,a,b,mu] = READ_UV
M = 943;
N = 1682;
D = 20;
fid = fopen('U.out');
U = fscanf(fid,'%f',[D,M]);
fclose(fid);

fid = fopen('V.out');
V = fscanf(fid,'%f',[D,N]);
fclose(fid);

fid = fopen('a.out');
a = fscanf(fid,'%f',[M,1]);
fclose(fid);


fid = fopen('b.out');
b = fscanf(fid,'%f',[N,1]);
fclose(fid);

fid = fopen('mu.out');
mu = fscanf(fid,'%f');
fclose(fid);
end