function [BIC,AIC] = bicCluster(X,Y)
% [BIC,AIC] = bicCluster(X,Y)
%
% Returns BIC and AIC criteria for a clustering solution. BIC (or AIC) is
% being computed based on the log-likelihood distance, which in this case,
% is determined using within-cluster variances. Like the Calinski-Harabasaz
% index, BIC and AIC penalize larger clustering solutions, which are
% expected to fit the data better just on the basis of approximating it.
%
% Inputs:
% X is data with N observations x P variables
% Y is the cluster associated with each variable (1 x P)
%
% This approach for adapting BIC to clustering solutions follows SPSS's
% implementation here:http://www.ibm.com/support/knowledgecenter/en/SSLVMB_22.0.0/com.ibm.spss.statistics.algorithms/alg_twostep.htm?view=embed
%
% May 13, 2019 // Alex teghipco (alex.teghipco@uci.edu) 

%% 1) Compute Nc vector showing number of objects per cluster (1 x K)
YUn = unique(Y);
for i = 1:length(YUn)
    Nc(i,1) = length(find(Y == YUn(i)));
end

%% 2) Compute vector Vc containing variances per cluster (P x K)
for i = 1:size(X,2) % loop over variables
    %disp(num2str(i))
    for j = 1:size(Nc,1) % loop over clusters
        idx = find(Y == YUn(j)); % find all objects belonging to cluster
        Vc(i,j) = (var(X(idx,i)) / length(idx)); % variance in variables for cluster
    end
end

%% 3) Compute P x 1 column containing variances for the whole sample and propagate by K to get matrix V
for i = 1:size(X,2)
    %disp(num2str(i))
    Vo(i,1) = (var(X(:,i))/(size(X(:,i),1) - 1));
end
V = repmat(Vo,[1,length(YUn)]);

%% 4) Compute log-likelihood LL
tmp1 = Vc + V; % have to split LL computation across steps to avoid taking log of zero
idx = find(tmp1 == 0); % find zeros
tmp1(idx) = 1.9763e-323; % closest value to zero that matlab can represent
tmp1 = log(tmp1)/2; % log of zero is inf
LL = -Nc'.*sum(tmp1,1); % 1 X K

%% 5) Compute BIC
BIC = -2 * sum(LL,2) + 2*size(Nc,1)*size(X,2) * log(size(X,1));

%% 6) Compute AIC
AIC = -2 * sum(LL,2) + 4*size(Nc,1)*size(X,2);

