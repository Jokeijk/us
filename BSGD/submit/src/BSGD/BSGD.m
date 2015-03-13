function BSGD
rng(0);
test_par = 0;   % whether test the parameters

% read the data
[userID1,movieID1,rate1] = read_data('data.txt');
userID = unique(userID1);  M = numel(userID);
movieID= unique(movieID1); N = numel(movieID);

if test_par
   [lambda_best, iter_best] = test_par(userID1,movieID1,rate1,M,N); 
else
    iter_best = 100;
    lambda_best = 25;
end

%######################################
% final run
%######################################
maxiter = iter_best;
lambda  = lambda_best;
   eta  = 1e-2;
D = 20;

% create data sparse matrix
Y_s = [userID1 movieID1 rate1];

% do the Bias-SGD
[U,V,a,b,mu,Ein,~] = do_bsgd(maxiter,lambda,eta,D,M,N,Y_s);

% write out the result
fid = fopen('U.out','w+');
fprintf(fid,'%f',U);
fclose(fid);
fid = fopen('V.out','w+');
fprintf(fid,'%f',V);
fclose(fid);
fid = fopen('a.out','w+');
fprintf(fid,'%f',a);
fclose(fid);
fid = fopen('b.out','w+');
fprintf(fid,'%f',b);
fclose(fid);
fid = fopen('mu.out','w+');
fprintf(fid,'%f',mu);
fclose(fid);
fid = fopen('Ein.out','w+');
fprintf(fid,'%f',Ein);
fclose(fid);


save('BSGD_out');


end

% Bias-SGD
function [U,V,a,b,mu,Ein,Eout] = do_bsgd(maxiter,lambda,eta,D,M,N,Y_in,Y_out)
if nargin < 7
    error('do_bsgd: not enough input!');
end

if nargin < 8
    iftest = 0;
    Eout = NaN;
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
U = rand(D,M);
V = rand(D,N);
a = rand(M,1);
b = rand(N,1);
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





