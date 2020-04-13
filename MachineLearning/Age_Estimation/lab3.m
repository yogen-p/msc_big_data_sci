%% Information
% facial age estimation
% regression method: linear regression

%% Settings
clear;
clc;

% path 
database_path = './data_age.mat';
result_path = './results/';

% initial states
absTestErr = 0;
cs_number = 0;


% cumulative error level
err_level = 5;

%% Training 
load(database_path);

nTrain = length(trData.label); % number of training samples
nTest  = length(teData.label); % number of testing samples
xtrain = trData.feat; % feature
ytrain = trData.label; % labels

w_lr = regress(ytrain,xtrain);
   
%% Testing
xtest = teData.feat; % feature
ytest = teData.label; % labels

yhat_test = xtest * w_lr;

%% 4. Compute the MAE and CS value (with cumulative error level of 5) for linear regression 

all_errors = abs(yhat_test-ytest); % Computing all errors
mae_linreg = sum(all_errors)/size(ytest, 1);   % MAE
cs_linreg = sum(all_errors < err_level == 1)/ size(ytest, 1);  % CS

fprintf('MAE Lin-Reg = %f\n', mae_linreg);
fprintf('CS Lin-Reg = %f\n', cs_linreg);

%% 5. Generate a cumulative score (CS) vs. error level plot by varying the error level from 1 to 15. 
%  The plot should look at the one in the Week6 lecture slides

for i = 1:15
    % Computing CS for varying error level
    cs_linreg(i) = sum(all_errors < i == 1)/ size(ytest, 1);
end

plot (cs_linreg, 'g-*')

xlabel('Error Level(1-15)')
ylabel('Cumulative Score')

%% 6. Compute the MAE and CS value (with cumulative error level of 5) 
%  for both partial least square regression and the regression tree model by using the Matlab built in functions.

%% 6.1 Partial least square regression

% Learning the PLS Regression model
[XL,YL,XS,YS,beta,PCTVAR,MSE,stats] = plsregress(xtrain, ytrain, 10);

 % Predicting the ages
yhat_test_p = [ones(size(xtest,1),1) xtest]*beta;

all_errors_p = abs(yhat_test_p-ytest); % Computing all errors
mae_p = sum(all_errors_p)/size(ytest, 1);   % MAE
cs_p = sum(all_errors_p < err_level == 1)/ size(ytest, 1);  % CS

fprintf('\n');
fprintf('MAE PLS-Reg = %f\n', mae_p);
fprintf('CS PLS-Reg = %f\n', cs_p);

%% 6.2 Regression tree

% Learning the Regression tree model
w_rt = fitrtree(xtrain, ytrain);
yhat_test_rt = predict(w_rt, xtest); % Predicting the ages

all_errors_rt = abs(yhat_test_rt-ytest); % Computing all errors
mae_rt = sum(all_errors_rt)/size(ytest, 1);   % MAE
cs_rt = sum(all_errors_rt < err_level == 1)/ size(ytest, 1);  % CS

fprintf('\n');
fprintf('MAE Reg-Tree = %f\n', mae_rt);
fprintf('CS Reg-Tree = %f\n', cs_rt);

%% 7. Compute the MAE and CS value (with cumulative error level of 5) for Support Vector Regression by using LIBSVM toolbox

addpath(genpath('./libsvm'));

% Fiting model with command line parameters
svr = svmtrain(ytrain, xtrain, '-s 3 -t 0'); 
yhat_test_svr = svmpredict(ytest, xtest, svr); % Predicting the ages

all_errors_svr = abs(yhat_test_svr-ytest); % Computing all errors
mae_svr = sum(all_errors_svr)/size(ytest, 1);   % MAE
cs_svr = sum(all_errors_svr < err_level == 1)/ size(ytest, 1);  % CS

fprintf('\n');
fprintf('MAE SV-Reg = %f\n', mae_svr);
fprintf('CS SV-Reg = %f\n', cs_svr);

