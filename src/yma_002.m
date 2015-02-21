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

templ= templateTree('MinLeaf',50);

% test the minimal leaf size
ntree = [10 100];
N = numel(ntree);
err = zeros(N,1);
for n=1:N
    cv = fitensemble(x_train, y_train, 'AdaBoostM1', ntree(n),templ,...
        'type','classification','kfold',5);
    err(n) = kfoldLoss(cv);
end

plot(ntree,err);
xlabel('Number of trees');
ylabel('Cross-validation error');

% use the optimal leaf size
tree = fitensemble(x_train, y_train, 'Bag', 100,templ,...
        'type','classification');
%view(tree,'mode','graph');

% prediction
y = predict(tree,x_test);

% output
fid = fopen('ENS_001.csv','w+');
fprintf(fid,'Id,Prediction\n');
for i=1:length(y)
    fprintf(fid,'%d,%d\n',ID(i),y(i));
end
fclose(fid);

end