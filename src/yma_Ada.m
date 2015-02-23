% Decision tree
% ensemble method

function yma_Ada
format long
Nf = 500;
data_dir = '../data';

% read the training data
M = csvread([data_dir,'/kaggle_train_tf_idf.csv'],1);
%M = csvread([data_dir,'/kaggle_train_wc.csv'],1);
x_train = M(:,2:Nf+1);
y_train = M(:, end);

% read the test data
M = csvread([data_dir,'/kaggle_test_tf_idf.csv'],1);
%M = csvread([data_dir,'/kaggle_test_wc.csv'],1);

x_test = M(:,2:Nf+1);
ID = M(:,1);

% % test for the optimal parameters
% ntree = [1000,2000];
% N = numel(ntree);
% 
% LEAF = [200,500,1000,2000];
% M = numel(LEAF);
% 
% err = zeros(N,M);
% 
% [n,m]=ndgrid(ntree(:),LEAF(:));
% 
% parfor (i=1:M*N,12)
%     err(i)=get_cv_score(x_train,y_train,m(i),n(i));
%     disp([m(i),n(i),err(i)])
% end
% 
% disp(err);
% dlmwrite('tmp.err',err,'delimiter',' ');
% %%% end of the test


% prediction
templ = templateTree('MinLeaf',200);
cv = fitensemble(x_train, y_train,'AdaBoostM1', 2000, templ,...
    'type','classification','LearnRate',0.1);
y = predict(cv,x_test);

save('Ada_001_2000');

% output
fid = fopen('Ada_001_2000.csv','w+');
fprintf(fid,'Id,Prediction\n');
for i=1:length(y)
    fprintf(fid,'%d,%d\n',ID(i),y(i));
end
fclose(fid);

function r=get_cv_score(x_train,y_train,leaf,ntree)
templ = templateTree('MinLeaf',leaf);
cv = fitensemble(x_train, y_train,'AdaBoostM1', ntree, templ,...
    'type','classification','kfold',5);
r= kfoldLoss(cv);
