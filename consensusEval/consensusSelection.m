function [PAC,areaK,deltaK] = consensusSelection_v3(cMat,varargin)
% [PAC,areaK,deltaK] = consensusSelection(cMat,kRange,varargin)
% ------------------------------------------------------------------------
% Select a good clustering solution based on consensus across many
% monte-carlo style permutations (see consensusClustering.m).
% consensusSelection will produce a density plot, CDF curve, plot of the
% relative change in AUC for the CDF curve, and PAC curve for each
% consensus matrix. 
%
% Proportion of ambiguously clustered pairs (PAC) is arguably the best
% metric by which to decide on an appropriate k. PAC characterizes the
% flatness of the CDF curve in some range between 0 and 1. By default,
% these range values will be set to [0.1 0.9]. Range [0,1] will likely fail
% because some portion of the CDF will show a high slope, revealing either
% pairs that are never clustered together (i.e., have a consensus of 0) and
% pairs that are always clustered together (i.e., have a consensus of 1).
% Thus, clustering solutions that have fewer pairs that have inconsistent
% cluster assignments are better, and a low PAC should reflect this. The
% range is data-dependent so inspect your CDF curves and alter these as
% necessary. For more information see: "Critical limitations of consensus
% clustering in class discovery" by ?enbabao?lu et al. Published in 2014 in
% Scientific Reports.
%
% The density plot shows the proportion of pairs that have different
% consensus values. In an ideal case, consensus values should show a
% bimodal distribution with isolated peaks at consensus values of 0 and 1.
% This would suggest that observations are clustered together and apart
% consistently throughout all permutations of the data.
%
% The CDF curves show the empirical CDF for each density plot. NOTE: the
% colors in the legend here map onto the same colors in the density plot.
% For more information see: "Consensus Clustering: A Resampling-Based Method
% for Class Discovery and Visualization of Gene Expression Microarray Data"
% published in Machine Learning in 2003 by Monti and colleagues. 
%
% The AUC for the CDF curves shows the relative change in the area under
% the curve with each additional cluster. That is, the AUC for the first
% consensus matrix is the actual area under the curve for that solution,
% but the AUC for each subsequent consensus matrix is subtracted from the
% AUC of the immediately preceeding solution. For more information see:
% "Consensus Clustering: A Resampling-Based Method for Class Discovery and
% Visualization of Gene Expression Microarray Data" published in Machine
% Learning in 2003 by Monti and colleagues.
%
% Mandatory arguments: ----------------------------------------------------
%       'cMat' : an n x p x m matrix of symmetrical consensus matrices of
%       size n x p for m clustering solutions. Should be in range [0,1] and
%       each additional consensus matrix should represent a clustering
%       solution with a larger number of clusters than the one preceeding
%       it. 
%
% Optional arguments: -----------------------------------------------------
%       'kRange' : an l x 1 vector where each element represents the number
%       of clusters for each consensus matrix m. For example, l(1) should
%       give the number of clusters evaluated in consensus matrix
%       cMat(:,:,1) and so on. 
%
%       'cdf' : 'true' will generate a CDF figure
%
%       'dist' : if 'true' will generate a distribution figure
%
%       'pac' : if 'true' will generate a PAC figure
%
%       'outDir' : output directory for figures
%
%       'numBins': this is the number of bins used for the density plot.
%       NOTE: the number of bins directly affects the CDF curve and
%       indirectly affects the way in which the AUC curve is computed.
%       Default is set to 100 as suggested here: https://bioconductor.org/packages/release/bioc/vignettes/ConsensusClusterPlus/inst/doc/ConsensusClusterPlus.pdf
%
%       'minC': this is the lower range used for the PAC
%       calculation.
%
%       'maxC': this is the upper range used for the PAC
%       calculation.
%
%       'cMap': r x 3 matrix of colors saved in a .mat file. Default is set
%       to 'distinguishable_colors'. If you have distinguishable_colors.m,
%       this script will be used to generate a unique number of colors. r
%       should correspond to the number of clustering solutions, otherwise
%       only the first k colors will be selected, where k corresponds to
%       the number of clustering solutions. If not using default cMap must
%       be a file path to a .mat file containing your r x 3 matrix. The
%       matrix may be of type single or double, or it can be a single or
%       double matrix contained in a structure. In case of the latter, the
%       matrix of colors should be contained in the very first field of the
%       structure. 
%
% Outputs: ----------------------------------------------------------------
%       'areaK' : AUC for the CDF curve of each consensus matrix. 
% 
%       'deltaK' : relative change in AUC for the CDF curve of each
%       consensus matrix. deltaK(1) = areaK(1) and deltaK(2) = areaK(2) -
%       areaK(1) and so on.
%
%       'ecdf_all' : empirical cdf for each consensus matrix. 
%
%       'consensus_all' : bins of consensus values corresponding to
%       values in 'ecdf_all' and 'count_all'.
%       
%       'count_all' : number of silhouettes in consensus matrix falling
%       into bins specified in 'consensus_all'.
%
%       'density_all' : density of silhouettes in consensus matrix falling
%       into bins specified in 'density_consensus'.
%
%       'density_consensus' : bins of consensus values corresponding to
%       values in 'density_all'. NOTE: the reason this differs from
%       'consensus_all' is because true density values will fall to the
%       left of 0 and the right of 1 even though no data points should lie
%       there. In the plot, we simply cut off the density at 0 and 1. 
% 
% Requires: ---------------------------------------------------------------
% Default options rely on distinguishable_colors.m for generating
% perceptually distinct colors.
%
% Alex Teghipco // alex.teghipco@uci.edu

% DEFAULT OPTIONS
histType = 'histfit'; % either 'ksdensity' or 'histfit' -- this will determine how distributions are generated

% Start reading in user-supplied inputs
options = struct('numBins',100,'minC',0.1,'maxC',0.9,'cMap','distinguishable_colors','outDir',[],'kRange',[],'cdf','true','dist','true','pac','true','auc','true','outAppend',[]);

% Read in the acceptable argument names
optionNames = fieldnames(options);

% Check the number of arguments passed
nArgs = length(varargin);
if round(nArgs/2)~=nArgs/2
    error('You are missing an argument name somewhere in your list of inputs')
end

% Assign supplied values to each argument in input
for pair = reshape(varargin,2,[]) %pair is {propName;propValue}
    inpName = pair{1}; % make case insensitive by using lower() here but this can be buggy
    if any(strcmp(inpName,optionNames))
        options.(inpName) = pair{2};
    else
        error('%s is not a recognized parameter name',inpName)
    end
end

if isempty(options.kRange)
   options.kRange = 1:size(cMat,3); 
end

% generate some distinguishable colors for each consensus matrix
if strcmp(options.cMap,'distinguishable_colors')
    cLines = distinguishable_colors(size(cMat,3));
else
    cLinesTmp = load(options.cMap);
    if isstruct(cLinesTmp)
        f = fieldnames(cLinesTmp);
        cLinesTmp = cLinesTmp.(f{1});
        cLines = cLinesTmp(1:length(options.kRange),:);
    elseif isa(cLinesTmp,'double') ||  isa(cLinesTmp,'single')
        cLines = cLinesTmp(1:length(options.kRange),:);
    end
end

%% 1) Make histograms
if strcmp(options.dist,'true')
    for clusti = 1:size(cMat,3)
        % vectorize upper triangle off current solution
        kMat = cMat(:,:,clusti).';
        m  = tril(true(size(kMat)),-1); % get lower triangle and exclude diagonal
        kMat  = kMat(m).';
        disp(['Working on histogram for cluster ' num2str(clusti) ' of ' num2str(size(cMat,3))]);
        
        % 1) Make density histograms
        if clusti == 1
            f1 = figure('Renderer', 'painters', 'Position', [10 10 1200 900]);
            switch histType
                case 'histfit'
                    h = histfit(kMat,options.numBins,'kernel');
                case 'ksdensity'
                    [f,xi] = ksdensity(kMat);
                    p = plot(xi,f,'r-','LineWidth',2);
            end
            hold on
        else
            figure(f1)
            switch histType
                case 'histfit'
                    h = histfit(kMat,options.numBins,'kernel');
                case 'ksdensity'
                    [f,xi] = ksdensity(kMat);
                    p = plot(xi,f,'r-','LineWidth',2);
            end
        end
        
        switch histType
            case 'histfit'
                h(1).FaceAlpha = 0;
                h(1).EdgeAlpha = 0;
                h(2).LineWidth = 3;
                h(2).Color = cLines(clusti,:);
                % compute empirical CDF for histogram
                %ecdf = cumsum(h(1).YData)/sum(h(1).YData);
            case 'ksdensity'
                p.LineWidth = 3;
                p.Color = cLines(clusti,:);
                % compute empirical CDF for histogram
                %ecdf = cumsum(p.YData)/sum(p.YData);
        end
    end
    a = get(gca,'Children');
    xdata = get(a, 'XData');
    ydata = get(a, 'YData');
        
    toRemove = find(mod(1:length(xdata),2) == 0);
    ydata(toRemove) = [];
    for i = 1:length(ydata)
        yMax(i,1) = max(ydata{i});
    end
    
    xlim([0 1]);
    ylim([0 max(yMax)])
    
    set(gcf,'color','w')
    set(gca,'FontSize',16)
    box off
    set(gca,'TickLength',[0 0])
    set(gca,'linewidth',3)
    title(['Histogram of each solution''s consensus matrix -- ' options.outAppend],'fontsize',22)
    ylabel('Density','fontsize',22)
    xlabel('Consensus','fontsize',22)
    if ~isempty(options.outDir)
        saveas(f1, [options.outDir '/Density' options.outAppend '.fig']);
        saveas(f1, [options.outDir '/Density' options.outAppend '.pdf']);
    end
    
    % insert new legend approach here...
    
    
end
hold off


%% 2) Get CDF 
for clusti = 1:size(cMat,3)
    % vectorize upper triangle off current solution
    kMat = cMat(:,:,clusti).';
    m  = tril(true(size(kMat)),-1); % get lower triangle and exclude diagonal
    kMat  = kMat(m).';
    [f{clusti},x{clusti}] = ecdf(kMat(:));
end

%% if you want to avoid using matlab's ecdf, just count...this is an example
% figure
% x = 0:0.01:1;
% for i = 1:29
%     tmp3 = (conFix(:,:,i));
%     for j = 1:length(x)
%         if j ~= length(x) 
%             %id = find(tmp3 >= x(j) & tmp3 <= x(j+1));
%             id = find(tmp3 <= x(j));
%             y2(j) = length(id) / (2335*2335);
%         else
%             %id = find(tmp3 >= x(j) & tmp3 <= 1);
%             id = find(tmp3 <= x(j));
%             y2(j) = length(id) / (2335*2335);
%         end
%     end
%     
%     p = plot((x),(y2));
%     if i == 1
%         hold on
%     end
%     p.Color = c(i,:);
%     p.LineWidth = 2;
% end


%% 3) Generate CDF figures
if strcmp(options.cdf,'true')
    for clusti = 1:size(cMat,3)
        if clusti == 1
            f2 = figure('Renderer', 'painters', 'Position', [10 10 1200 900]);
            p = plot(x{clusti},f{clusti});
            hold on
        else
            p = plot(x{clusti},f{clusti});
        end
        p.LineWidth = 3;
        p.Color = cLines(clusti,:);
    end
    hold off
    set(gcf,'color','w')
    set(gca,'FontSize',16)
    box off
    set(gca,'TickLength',[0 0])
    set(gca,'linewidth',3)
    for clusti = 1:size(cMat,3)
        lNames{clusti} = ['k = ' num2str(options.kRange(clusti))];
    end
    [~, hobj, ~, ~] = legend(lNames,'location','best','Fontsize',16);
    hl = findobj(hobj,'type','line');
    set(hl,'LineWidth',8);
    
    title(['CDF -- ' options.outAppend],'fontsize',22)
    ylabel('CDF','fontsize',22)
    xlabel('consensus','fontsize',22)
    
    if ~isempty(options.outDir)
        saveas(f2, [options.outDir '/CDF' options.outAppend '.fig']);
        saveas(f2, [options.outDir '/CDF' options.outAppend '.pdf']);
    end
end

%% 4) Compute change in the area under each CDF curve
if strcmp(options.dist,'true') && strcmp(options.auc,'true') && size(cMat,3) > 1
    for clusti = 1:size(cMat,3)
        kMat = cMat(:,:,clusti).';
        m  = tril(true(size(kMat)),-1); % get lower triangle and exclude diagonal
        kMat  = kMat(m).';
        
        % This did not work, but was taken from an R package: https://rdrr.io/bioc/ConsensusClusterPlus/src/R/ConsensusClusterPlus.R
        %currArea = 0;
        %for i = 1:length(ecdf)-1
        %    currArea = currArea + ecdf(i)*(ecdf(i+1)-ecdf(i));
        %end
        %areaK(clusti,1) = currArea;
        
        % This worked, just compute the actual AUC the normal way
        %areaK_v2(clusti,1) = trapz(h(1).XData,ecdf);
        
        % But this is the implementation of the heuristic from the
        % original consensus clustering paper so we will use it. It
        % approximates the actual AUC surprisingly well
        skMat = sort(kMat,'ascend');
        for j = 2:length(skMat)
            switch histType
                case 'histfit'
                    idx = find(h(1).XData > skMat(j));
                case 'ksdensity'
                    idx = find(p.XData > skMat(j));
            end
            if ~isempty(idx)
                tmpSk(j-1,1) = (skMat(j) - skMat(j-1))*f{clusti}(idx(1));
            else
                tmpSk(j-1,1) = 0;
            end
        end
        currArea = sum(tmpSk);
        clear tmpSk
        areaK(clusti,1) = currArea;
        
        % get change in auc -- delta k
        if clusti == 1
            deltaK(clusti,1) = areaK(clusti);
        else
            deltaK(clusti,1) = (areaK(clusti) - areaK(clusti-1))/areaK(clusti-1);
        end
        
    end
    
    % now make a delta area figure
    f3 = figure('Renderer', 'painters', 'Position', [10 10 1200 900]);
    dk = plot(kRange(2:end),deltaK,'LineStyle','--','Marker','o','Color',[0.2 0.2 0.2],'MarkerFaceColor',[1 1 1],'LineWidth',3,'MarkerSize',10);
    set(gcf,'color','w')
    set(gca,'FontSize',16)
    set(gca,'linewidth',3)
    box off
    set(gca,'TickLength',[0 0])
    
    title(['Relative change in area under the curve -- ' options.outAppend],'fontsize',22)
    ylabel('delta AUC','fontsize',22)
    xlabel('k','fontsize',22)
    
    if ~isempty(options.outDir)
        saveas(f3, [options.outDir '/DeltaAUC' options.outAppend '.fig']);
        saveas(f3, [options.outDir '/DeltaAUC' options.outAppend '.pdf']);
    end
else
    deltaK = [];
    areaK = [];
end

%% 5) Calculate PAC
for clusti = 1:size(cMat,3)
    try
        xId1 = find(x{clusti}>=  options.maxC);
        if isempty(xId1)
            xId1 = length(x);
        else
            xId1 = xId1(1);
        end
        
        xId2 = find(x{clusti}<=  options.minC);
        if isempty(xId2)
            xId2 = 1;
        else
            xId2 = xId2(end);
        end
        
        PAC(clusti,1) = f{clusti}(xId1) - f{clusti}(xId2);
    catch
        PAC(clusti,1) = NaN;
    end
end
if strcmp(options.pac,'true')
    f4 = figure('Renderer', 'painters', 'Position', [10 10 1200 900]);
    p = plot(options.kRange,PAC,'LineStyle','--','Marker','o','Color',[0.2 0.2 0.2],'MarkerFaceColor',[1 1 1],'LineWidth',3,'MarkerSize',10);
    set(gcf,'color','w')
    set(gca,'FontSize',16)
    set(gca,'linewidth',3)
    box off
    set(gca,'TickLength',[0 0])
    
    title(['Proportion of ambiguously clustered pairs across solutions -- ' options.outAppend],'fontsize',22)
    ylabel('PAC','fontsize',22)
    xlabel('k','fontsize',22)
    
    if ~isempty(options.outDir)
        saveas(f4, [options.outDir '/PAC' options.outAppend '.fig']);
        saveas(f4, [options.outDir '/PAC' options.outAppend '.pdf']);
    end
end
