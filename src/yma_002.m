% Decision tree
% ensemble method

function ENS_001
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

ntree = [200 400 800 1200]; 
N = numel(ntree);

nleaf = [50,100,200,400,800,1200,2000];
M = numel(nleaf);

err = zeros(N,M);
for n=1:N
    for m=1:M
        templ= templateTree('MinLeaf',nleaf(m));
        cv = fitensemble(x_train, y_train, 'AdaBoostM1', ntree(n),templ,...
            'type','classification','KFold',10);
        err(n,m) = kfoldLoss(cv);
    end
end

dlmwrite('tmp.err',err,'delimiter',' ');
pause;

% use the optimal leaf size
tree = fitensemble(x_train, y_train, 'Bag', 100,templ,...
        'type','classification');
%view(tree,'mode','graph');

% prediction
y = predict(tree,x_test);

% output
fid = fopen('ENS_002.csv','w+');
fprintf(fid,'Id,Prediction\n');
for i=1:length(y)
    fprintf(fid,'%d,%d\n',ID(i),y(i));
end
fclose(fid);

end