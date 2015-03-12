function BSGD
rng(0);
test_par = 0;   % whether test the parameters

% read the data
[userID1,movieID1,rate1] = read_data('data.txt');
userID = unique(userID1);  M = numel(userID);
movieID= unique(movieID1); N = numel(movieID);

if test_par
    
    % 5 fold cross-validation
    Ndata = numel(rate1);
    nf    = floor(Ndata/5);
    order = randperm(Ndata);
    
    % parameter
    maxiter = 100;
    lambda  = 10:5:40;
    eta  = 1e-2;
    D = 20;
    
    % test for lambda
    Ein_all  = zeros(maxiter, length(lambda));
    Eout_all = zeros(maxiter, length(lambda));
    for i = 1:length(lambda)
        for k = 1
            out_id = order( (k-1)*nf+1 : k*nf );
            in_id  = 1:Ndata;
            in_id(out_id) = [];
            
            Y_in  = [userID1(in_id) movieID1(in_id),rate1(in_id)];
            Y_out = [userID1(out_id), movieID1(out_id),rate1(out_id)];
            
            [~,~,~,~,~,Ein,Eout] = do_bsgd(maxiter,lambda(i),eta,D,M,N,Y_in,Y_out);
        end
        Ein_all(:,i)  = Ein;
        Eout_all(:,i) = Eout;
    end
    % write out the result
    dlmwrite('Ein_lambda.txt',Ein_all,'delimiter',' ','precision',8);
    dlmwrite('Eout_lambda.txt',Eout_all,'delimiter',' ','precision',8);
    
    [Ein_min,~]  = min(Ein_all,[],1);
    [Eout_min,min_id] = min(Eout_all,[],1);
    
    % find best lamda
    [Eout_best,best_id] = min(Eout_min);
    lambda_best = lambda(best_id);
    
    % find best iteration number
    iter_best = min_id(best_id);
    
    fprintf('Best lamda=%f, Best iter=%d\n',lambda_best,iter_best);
    % plot the result
    figure(1);
    % plot E vs lambda
    subplot(121)
    plot(lambda,Ein_min,'ro-');
    hold on;
    plot(lambda,Eout_min,'bo-');
    plot(lambda_best, Eout_best,'b*');
    xlabel('\lambda');
    ylabel('RMSE');
    legend('Ein','Eout');
    hold off;
    set(gca,'fontsize',13)
    grid on;box on;
    hold off;
    
    % plot E vs iter for best lambda
    subplot(122)
    plot(1:maxiter,Ein_all(:,best_id),'r-');
    hold on;
    plot(1:maxiter,Eout_all(:,best_id),'b-');
    plot(iter_best,Eout_best,'b*');
    xlabel('# of iteration');
    ylabel('RMSE');
    legend('Ein','Eout');
    text(5,1.04,['\lambda=',num2str(lambda_best)],'fontsize',13);
    set(gca,'fontsize',13)
    grid on; box on;
    hold off;
    
    
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





