function [lambda_best, iter_best] = test_par(userID1,movieID1,rate1,M,N) 

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