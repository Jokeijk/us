function BSGD
rng(0);

% read the data
[userID1,movieID1,rate1] = read_data('data.txt');
userID = unique(userID1);
[movieID,movieName,movieGenre,Genres] = read_movie('movies.txt');

% 5 fold cross-validation
Ndata = numel(rate1);
nf    = floor(Ndata/5);
order = randperm(Ndata); 

% parameter
maxiter = 300;
lambda  = [1e-1,1,10,100];
   eta  = 1e-2;
D = 1;


% test for lambda
Ein_all  = zeros(maxiter, length(lambda));
Eout_all = zeros(maxiter, length(lambda));
for i = 1:length(lambda)
    for k = 1
        out_id = order( (k-1)*nf+1 : k*nf );
        in_id  = 1:Ndata;
        in_id(out_id) = [];
        
        [M,N,Y_in]  = sparse_matrix(userID, movieID, userID1(in_id), movieID1(in_id),rate1(in_id));
        [~,~,Y_out] = sparse_matrix(userID, movieID, userID1(out_id), movieID1(out_id),rate1(out_id));
        
        [~,~,~,~,~,Ein,Eout] = do_bsgd(maxiter,lambda(i),eta,D,M,N,Y_in,Y_out);
    end
    Ein_all(:,i)  = Ein;
    Eout_all(:,i) = Eout;
end
% write out the result
dlmwrite('Ein_lambda.txt',Ein_all,'delimiter',' ','precision',8);
dlmwrite('Eout_lambda.txt',Eout_all,'delimiter',' ','precision',8);

figure(1)
subplot(121)
plot(1:maxiter,Ein_all);
title('Ein');
subplot(122)
plot(1:maxiter,Eout_all);
title('Eout');


pause;





% Final run, the paramter has been checked with cross-validation
% parameters
maxiter = 200;
lambda  = 1;
   eta  = 1e-2;
k = 20;

% create data sparse matrix
[M,N,Y_s] = sparse_matrix(userID, movieID, userID1, movieID1,rate1);
% do the Bias-SGD
[U,V,a,b,mu,Ein,Eout] = do_bsgd(maxiter,lambda,eta,k,M,N,Y_s);


pause;
end

% Bias-SGD
function [U,V,a,b,mu,Ein,Eout] = do_bsgd(maxiter,lambda,eta,D,M,N,Y_in,Y_out)
if nargin < 7
    error('do_bsgd: not enough input!');
end

if nargin < 8
    iftest = 0;
else
    iftest = 1;
end

Ndata = size(Y_in,1);
Ein   = zeros(maxiter,1);
if iftest
    Ntest = size(Y_out,1);
    Eout  = zeros(maxiter,1);
end

% initialization
U = zeros(D,M) + 1;
V = zeros(D,N) + 1;
a = zeros(M,1);
b = zeros(N,1);
mu= mean(Y_in(:,3));

% start iteration
for i = 1:maxiter
    order = randperm(Ndata);
    % loop through the data
    for j = 1:Ndata
        ii = Y_in(order(j),1);
        jj = Y_in(order(j),2);
        yy = Y_in(order(j),3);
        err= U(:,ii)'* V(:,jj) + a(ii) + b(jj) + mu - yy;
        
        U(:,ii) = U(:,ii) - 2*eta*err*V(:,jj);
        V(:,jj) = V(:,jj) - 2*eta*err*U(:,ii);
        a(ii)   = a(ii) - 2*eta*err;
        b(jj)   = b(jj) - 2*eta*err;
    end % end of loop
    
    % correct for damping term
    U = U * (1-eta*lambda);
    V = V * (1-eta*lambda);
    a = a * (1-eta*lambda);
    b = b * (1-eta*lambda); 
    
    % RMS error
    Ein(i) = calc_RMS(Ndata,Y_in,U,V,a,b,mu);
    if iftest
        Eout(i) = calc_RMS(Ntest,Y_out,U,V,a,b,mu);
        fprintf('iter=%5d  RMSEin=%10.5f RMSEout=%10.5f\n',i,Ein(i),Eout(i));
    else
        fprintf('iter=%5d  RMSEin=%10.5f\n',i,Ein(i));
    end
    
%     if i>1 && Eout(i)>Eout(i-1)
%         Eout(i+1:end) = NaN;
%         Ein(i+1:end)  = NaN;
%         break;
%     end
    
    % decrease the step size
    eta = eta * 0.9;
    %eta = max(eta * 0.9, 2e-4);
end % end of iteration
end

% Calculate the RMS error
function RMSE = calc_RMS( N, Y, U, V, a, b, mu)
RMSE = 0;
for j = 1:N
    ii = Y(j,1);
    jj = Y(j,2);
    yy = Y(j,3);
    err= U(:,ii)'* V(:,jj) + a(ii) + b(jj) + mu - yy;
    RMSE = err^2 + RMSE;
end
RMSE = sqrt(RMSE/N);

end





