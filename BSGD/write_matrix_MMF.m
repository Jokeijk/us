function write_matrix_MMF(filename,M,N,Y_s)
a = date;
fid = fopen(filename,'w+');
fprintf(fid,'%%%%MatrixMarket matrix coordinate real general\n');
fprintf(fid,'%% %s\n',a);
fprintf(fid,'%d %d %d\n',M,N,size(Y_s,1));
for i=1:size(Y_s)
    fprintf(fid,'%6d %6d %5f\n',Y_s(i,1),Y_s(i,2),Y_s(i,3));
end
fclose(fid);