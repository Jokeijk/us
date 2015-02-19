% Decision tree

function DCT_001
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

% % test the minimal leaf size
% leafs = logspace(1,1.8,4);
% N = numel(leafs);
% err = zeros(N,1);
% for n=1:N
%     t = fitctree(x_train, y_train, 'CrossVal','On',...,
%         'MinLeaf',leafs(n));
%     err(n) = kfoldLoss(t);
% end
% 
% plot(leafs,err);
% xlabel('Min Leaf Size');
% ylabel('Cross-validated error');

% use the optimal leaf size
OptimalTree = fitctree(x_train,y_train,'minleaf',18);
view(OptimalTree,'mode','graph');

[~,~,~,bestlevel] = cvLoss(OptimalTree,...
    'SubTrees','All','TreeSize','min');
tree = prune(OptimalTree,'Level',bestlevel);
view(tree,'mode','graph');
% prediction
y = predict(tree,x_test);

% output
fid = fopen('DCT_002.csv','w+');
fprintf(fid,'Id,Prediction\n');
for i=1:length(y)
    fprintf(fid,'%d,%d\n',ID(i),y(i));
end
fclose(fid);

end