function [clusterConsensus,itemTracker,clusterTracker,exemplarTracker] = consensusClustering_v4(inData,nPerms,pItem,kRange,cAlgo,distance,neighbor_num,maxIter,maxReps,center,scale)

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
    center = 'true';
end
if isempty(scale)
    scale = 'true';
end
verbose = '100'; % or 'false'; or '100' for every 100
scThresh = 0.2;

% make sure rng is truly random
rng('default');
rng('shuffle');
affinFixedK = 'false';
affinAdapt = 'true';

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
    autoData = inData(subIdxS,:);
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
                 %splot = 'plot'; % observing a clustering process when it is on
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
                 for z = 1:length(kRange)
                     [idx,netsim,dpsim,expref,pref] = apclusterK(autoData,k,0);
                     idxU = unique(idx);
                     for l = 1:length(idxU)
                         cluster(:,z) = find(idx == idxU(l));
                     end
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
                         clusters(:,z) = kmeans_v2(autoData,kRange(z),'MaxIter',maxIter,'Distance','correlation','Replicates',maxReps); % was 100
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
                         clusters(:,z) = kmedoids(autoData,kRange(z),'options',opts,'Distance',@etaSquared2,'Replicates',maxReps,'Algorithm','pam');
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
        case {'kmeans';'PAM';'spectral'}
            for i = 1:size(clusters,2) % for each solution
                for m = 1:kRange(i)
                    id = find(clusters(:,i) == m);
                    clusterTracker(subIdxS(id),subIdxS(id),i) = clusterTracker(subIdxS(id),subIdxS(id),i) + 1;
                end
            end
            clear clusters
        case {'affinitypropagation'}
            
    end
end

switch cAlgo
    case {'mod'}
        for i = 1:size(clusterTracker,3)
            clusterConsensus(:,:,1) = clusterTracker(:,:,1)./itemTracker(:,:,1);
        end
        clusterTracker = clusterTracker(:,:,1);
        itemTracker = itemTracker(:,:,1);
        
    case {'kmeans'; 'PAM'; 'spectral'}
        for i = 1:size(clusterTracker,3)
            clusterConsensus(:,:,i) = clusterTracker(:,:,i)./itemTracker(:,:,1);
        end
    case {'affinitypropagation'}
        
        
end



