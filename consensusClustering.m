function [clusterConsensus,itemTracker,clusterTracker,exemplarTracker, alignedC] = consensusClustering(inData,nPerms,pItem,kRange,cAlgo,distance,neighbor_num,maxIter,maxReps,center,scale,verbose)
% Perform consensus clustering using a variety of clustering algorithms.
% Please ensure the algorithm of your choice (as contained in this package
% is in your matlab path)
%
% Inputs ---------------------------------------------------
% inData: a matrix of n samples by p features to cluster (we are clustering
%   samples)
% nPerms: number of subsamples to draw from inData and cluster to generate
%   the co-association or consensus matrix. (default: 7500)
% pItem: proportion of samples you want to subsample (default: 0.8)
% kRange: a vector of classes you want in each independently constructed
%   solution. For example: [2:10] will produce 9 consensus matrices for
%   solutions containing between 2 and 10 clusters.
% cAlgo: clustering algorithm. Options are:
%   'spectral': self-supervised spectral clustering
%    'mod': Louvain clustering (modularity based)
%    'affinitypropagation': affinity propgataion (information-theory
%    based). Note we will use a version of AP that seeks a solution with a
%    specific k. Change first few lines of code to not do this but you
%    might break things. 
%    'PAM': k-medoids using the PAM algorithm
%    'kmeans': k-means++ clustering (if eta2 is used we will draw on a
%    bespoke version of k-means instead of matlab default)
%   (default: 'affinitypropagation')
% distance: distance metric to use. Options are:
%   'r': Pearson correlation coefficient
%   'eta': Cohen et al's eta squared measure of similarity for whole brain
%       connectivity maps
%   'seuclidean': euclidean distance
%   'squaredeuclidean': squared euclidean distances
%   'cosine': cosine distance
%   'raw': use a precomputed distance matrix (i.e., inData is actually n x
%    n instead of n x p. (default: 'euclidean'). NOTE, this will *NOT* work
%    for k-means or k-medoids.
% 'neighbor_num': number of neighbors for spectral clustering. Default: 100 
% 'maxIter': maximum iterations of clustering algorithm (applies to all)
%   (default: 10000)
% 'maxReps': maximum replicates of clustering algorithm (applies to k-means
%   and PAM) (default: 15)
% 'center': set to true to center input matrix inData
% 'scale': set to true to scale input matrix inData
% 'verbose': set to 'true' or 'false'; or '100' for every 100
%
% Outputs --------------------------------------------
% clusterConsensus: consensus matrix of n x n x s where s is each solution
%   specified by kRange
% itemTracker: n x n matrix showing proportion of times a pair of samples were subsampled together 
% clusterTracker: n x n matrix showing proportion of times a apair of
%   samples were placed in the same cluster
% exemplarTracker: n x s matrix showing number of times a sample was chosen
%   as an examplar for a cluster
% alignedC: aligned cluster identities across all solutions and for each
% subsample (i.e., n x k x subsample).
%
%
% Important -------------------------------------------
% 1) Do not use a precomputed matrix with an algorithm that does not
% support it! 
% 2) For affinity propagation we search for k. This is inefficient and you
% may need to increase damping factor, and tune some parameters if you are
% seeing warnings
% 3) Both affinity propagation and Louvain clustering do not technically
% require an a priori defined k. This can make consensus clustering less
% straightforward. Also, note affinity propagation used in this way has not
% beed debugged.
% 4) Technically, you should be able to use adaptive AP with a fixed k but
% in my experience this has not worked very well (the adaptive AP package
% was not really built with this in mind). 

% Missing; kmeans_v2.m

alignedC = [];
if isempty(pItem)
    pItem = 0.8;
end
if isempty(kRange)
    kRange = 2:size(inData,2);
end
if isempty(cAlgo)
    cAlgo = 'affinitypropagation';
end
if isempty(nPerms)
    nPerms = 7500;
end
if isempty(neighbor_num)
    neighbor_num = 100;
end
if isempty(distance)
    distance = 'euclidean';
end
if isempty(maxIter)
    maxIter = 10000;
end
if isempty(maxReps)
    maxReps = 15;
end
if isempty(center)
    center = 'false';
end
if isempty(scale)
    scale = 'false';
end
if isempty(verbose)
    verbose = 'true'; % or 'false'; or '100' for every 100
end

% add to path
fp = which('consensusClustering');
[fp,~,~] = fileparts(fp);
addpath(genpath(tmp1))

% make sure rng is truly random
rng('default')
rng('shuffle');
affinFixedK = 'true';
affinAdapt = 'false';

% get sample size
itemsPerResample = round((size(inData,1))*pItem);
disp(['Actual percentage of items per subsample is ' num2str(itemsPerResample/(size(inData,1)))])

% set up matrices for tracking number of times items are grouped together
% and subsampled together
itemTracker = zeros([size(inData,1),size(inData,1),length(kRange)]);
clusterTracker = zeros([size(inData,1),size(inData,1),length(kRange)]);
exemplarTracker = zeros([size(inData,1),length(kRange)]);

for j = 1:nPerms
    switch verbose
        case 'true'
            disp(['Working on subsample ' num2str(j) ' of ' num2str(nPerms)])
        case '100'
            if intersect(j,[1:100:nPerms])
                disp(['Working on subsample ' num2str(j) ' of ' num2str(nPerms)])
            end
    end
    % take a subsample of items
    subIdx = randperm(size(inData,1),itemsPerResample);
    subIdxS = sort(subIdx);
    itemTracker(subIdxS,subIdxS) = itemTracker(subIdxS,subIdxS) + 1;
    if strcmpi(distance,'raw')
        if ~strcmpi(cAlgo)
            autoData = inData(subIdxS,subIdxS);
        end
    else
        autoData = inData(subIdxS,:);
    end
    
    % scale and center input data
    if strcmp(center,'true')
        autoData = autoData - repmat(mean(autoData),size(autoData,1),1);
    end
    if strcmp(scale,'true')
        autoData = autoData/max(max(abs(autoData)));
    end
    
    % distance measure
    switch cAlgo
        case {'spectral';'mod';'affinitypropagation'}
            switch distance
                case 'r'
                    autoData = corr(autoData');
                    id = find(autoData < 0);
                    autoData(id) = 0;
                    autoData = 1 - autoData;
                case 'eta'
                    autoData = etaSquared2_fast(autoData,autoData,'distance','true');
                case 'euclidean'
                    autoData = dist2(autoData,autoData);
                case 'seuclidean'
                    autoData = pdist2(autoData,autoData,'Distance','seuclidean');
                case 'squaredeuclidean'
                    autoData = pdist2(autoData,autoData,'Distance','squaredeuclidean');
                case 'cosine'
                    autoData = pdist2(autoData,autoData,'Distance','cosine');
            end
    end
        
     switch cAlgo
        case {'spectral'}
            clear A_LS ZERO_DIAG clusts_RLS rlsBestGroupIndex qualityRLS clusters clustTmp
            [D_LS,A_LS,LS] = scale_dist(double(autoData),floor(neighbor_num/2)); %% Locally scaled affinity matrix
            clear D_LS LS;
            remove = isnan(A_LS);
            idx = find(remove);
            A_LS(idx) = 0;
            % zero out diagonal
            ZERO_DIAG = ~eye(size(autoData,1));
            A_LS = A_LS.*ZERO_DIAG;
            % cluster all clustering choices
            [clusts_RLS, rlsBestGroupIndex, qualityRLS] = cluster_rotate(A_LS,kRange,0,1);
            for m = 1:length(clusts_RLS)
                clustTmp = clusts_RLS{m};
                clusters(:,m) = zeros([size(autoData,1), 1]);
                for l = 1:length(clustTmp)
                    clusters(clustTmp{l},m) = l;
                end
            end
         case {'mod'}
             switch distance
                 case {'euclidean'}
                     autoData = 1./(1 + autoData);
                     
                 case {'r';'eta'}
                     autoData = 1 - autoData;
             end
             
             [B,~] = modularity(double(autoData),1);
             [clusters,~] = genlouvain(B,maxIter,0);
         case {'affinitypropagation'}
             if strcmp(affinFixedK,'false') && strcmp(affinAdapt,'true')
                 algorithm = 1;  % 1 --- adaptive AP, 0 --- original AP
                 nrun = maxIter;   % max iteration times, default 50000
                 nconv = 50;     % convergence condition, default 50
                 pstep = 0.01;   % decreasing step of preferences: pstep*pmedian, default 0.01
                 lam = 0.5;      % damping factor, default 0.5
                 cut = 1;        % after clustering, drop an cluster with number of samples < cut
                 splot = 'noplot';
                 simatrix = 1;
                 
                 [clusters,~,~,~,~,~,~,~] = adapt_apcluster(autoData,[],...
                     [],pstep,simatrix,'convits',nconv,'maxits',nrun,'dampfact',lam,splot);
                 
             elseif strcmp(affinFixedK,'false') && strcmp(affinAdapt,'false')
                 [idx] = apcluster(autoData);
                 idxU = unique(idx);
                 for l = 1:length(idxU)
                     cluster = find(idx == idxU(l));
                     %clusterTracker(subIdxS(idx2),subIdxS(idx2),i) = clusterTracker(subIdxS(idx2),subIdxS(idx2),i) + 1;
                 end
             elseif strcmp(affinFixedK,'true') && strcmp(affinAdapt,'false')
                 parfor z = 1:length(kRange)
                     [idx{z},~,~,~,~] = apclusterK(autoData,kRange(z),0,4000,400);
                 end
                 for z = 1:length(kRange)
                     idxU = unique(idx{z});
                     for l = 1:length(idxU)
                         tmp = find(idx{z} == idxU(l));
                         clusters(tmp,z) = l;
                     end
                 end
                 %align clusters
                 for z = 1:length(kRange)
                     kkeep(:,j,z) = clusters;
                     kkeepR(:,j,z) = idx{z};
                     idxU = unique(idx{z});
                     exemplarTracker(idxU,z) = exemplarTracker(idxU,z)+1;
                 end
             end
     end
    
     switch cAlgo
         case {'kmeans'}
             switch distance
                 case {'euclidean'}
                     for z = 1:length(kRange)
                         clusters(:,z) = kmeans(autoData,kRange(z),'MaxIter',maxIter,'Distance','sqeuclidean','Replicates',maxReps); % was 100
                     end 
                 case {'standardeuclidean';'squaredeuclidean';'cosine'}
                     for z = 1:length(kRange)
                         clusters(:,z) = kmeans(autoData,kRange(z),'MaxIter',maxIter,'Distance',distance,'Replicates',maxReps); % was 100
                     end
                 case {'r'}
                     for z = 1:length(kRange)
                         clusters(:,z) = kmeans(autoData,kRange(z),'MaxIter',maxIter,'Distance','correlation','Replicates',maxReps); % was 100
                     end
                 case {'eta'}
                     for z = 1:length(kRange)
                         clusters(:,z) = kmeans_v2(autoData,kRange(z),'MaxIter',maxIter,'Distance','correlation','Replicates',maxReps,'Options',statset('UseParallel',1)); % was 100
                     end
             end
         case {'PAM'}
             opts.MaxIter = maxIter;
             switch distance
                 case {'r';'euclidean';'standardeuclidean';'squaredeuclidean';'cosine'}
                     for z = 1:length(kRange)
                         clusters(:,z) = kmedoids(autoData,kRange(z),'options',opts,'Distance',distance,'Replicates',maxReps,'Algorithm','pam');
                     end
                 case {'eta'}
                     for z = 1:length(kRange)
                         clusters(:,z) = kmedoids(autoData,kRange(z),'options',opts,'Distance',@etaSquared2_fast,'Replicates',maxReps,'Algorithm','pam');
                     end
             end
     end
     
    % how often is each pair of items placed in the same
    % cluster
    switch cAlgo
        case {'mod'}
            for i = 1:size(clusters,2) % for each solution
                for m = 1:max(clusters)
                    id = find(clusters(:,i) == m);
                    clusterTracker(subIdxS(id),subIdxS(id),i) = clusterTracker(subIdxS(id),subIdxS(id),i) + 1;
                end
            end
            clear clusters
        case {'kmeans';'PAM';'spectral';'affinitypropagation'}
            for i = 1:size(clusters,2) % for each solution
                mtmp = clusters(:,i);
                for m = 1:max(mtmp)
                    id = find(mtmp == m);
                    clusterTracker(subIdxS(id),subIdxS(id),i) = clusterTracker(subIdxS(id),subIdxS(id),i) + 1;
                end
            end
            clear clusters
    end
end

switch cAlgo
    case {'mod'}
        for i = 1:size(clusterTracker,3)
            clusterConsensus(:,:,i) = clusterTracker(:,:,1)./itemTracker(:,:,1);
        end
        clusterTracker = clusterTracker(:,:,i);
        itemTracker = itemTracker(:,:,1);
        
    case {'kmeans'; 'PAM'; 'spectral'; 'affinitypropagation'}
        for i = 1:size(clusterTracker,3)
            clusterConsensus(:,:,i) = clusterTracker(:,:,i)./itemTracker(:,:,1);
        end        
        if strcmpi(cAlgo,'affinitypropagation')
            exemplarTracker = exemplarTracker./nPerms;
            for j = 1:length(kRange)
                alignedC(:,:,j) = clusterAlignment(kkeep(:,:,j)','sortClusters','false','renameClusters','false','alignOnly','false')';
            end
        end
end



