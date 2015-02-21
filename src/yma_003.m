% Decision tree
% ensemble method

function ENS_002
Nf = 500;
data_dir = '../data';

% read the training data
M = csvread([data_dir,'/kaggle_train_tf_idf.csv'],1);
x_train = M(:,2:Nf+1);
y_train = M(:, end);

% read the test data
M = csvread([data_dir,'/kaggle_test_tf_idf.csv'],1);
x_test = M(:,2:Nf+1);
ID = M(:,1);

% test the minimal leaf size
ntree = 100;
N = numel(ntree);
err = zeros(N,1);
for n=1:N
    cv = TreeBagger(ntree(n),x_train, y_train,...
        'Method','classification',...
        'OOBPred','On','NVarToSample',300);
    tmp = oobError(cv);
    err(n) = tmp(end);
end

% plot(ntree,err);
% xlabel('Number of trees');
% ylabel('Cross-validation error');


% prediction
y = predict(cv,x_test);

% output
fid = fopen('ENS_001.csv','w+');
fprintf(fid,'Id,Prediction\n');
for i=1:length(y)
    fprintf(fid,'%d,%d\n',ID(i),str2num(y{i}));
end
fclose(fid);

end