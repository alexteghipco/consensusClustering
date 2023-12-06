function [locOpt,globOptPac,dP,postdPCorrR_pacbc,postdPCorrP_pacbc,postdPCorrR_pacdip,postdPCorrP_pacdip,predPCorrR_pacbc,predPCorrP_pacbc,predPCorrR_pacdip,predPCorrP_pacdip,sigmaTolNew] = optimalConsensus(p,bc,dip,dipP,dipCase,type,numStdDevs,pThresh,sigmaTol,kRange,autoAdjFac,conThresh,varargin)
% v -- either pac or binomial coefficient; n x p; p are subjects; n are
% solutionse
% type -- 'intersect' or 'globMax'
% sigmaTol -- 3;
% pThresh = 0.001;
% kRange = 2:30;
% numStdDevs = 1.5;
% dipCase = 'noiseEst';


options = struct('distance','false','smallest',[],'largest',[]);

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

autoSigmaAdj = 'true';
if isempty(conThresh)
    conThresh = 4;
end
noConsec = 'true';

for i = 1:size(p,2)
    % get current sub
    pTmp = p(:,i);
    dTmp = dip(:,i);
    dTmpP = dipP(:,i);
    bTmp = bc(:,i);
    % if everything is not zeros, continue...
    if sum(pTmp) ~= 0  % first check if everything is zero
        %% 1) Get plateau in dip and remove it from potential solutions
        switch dipCase
            case 'noise'
                d = abs(diff(dTmp)); % get absolute successive differences
                n = sigmaTol*sqrt(estimatenoise(d)); % 3 sigma tolerance; we take this approach because the plateau becomes centered around zero. This means we can do a simple test if we can estimate noise in the data (some sigma value)
                pbool = (d <= n); % find values below "noise"
                id = find(pbool == 1);
                
                % remove any id that is clearly out of place (5 indices
                % behind a consecutive string by default)
                % -- find all non-consecutive values (gaps)
                
                % old version
                %                 if ~isempty(id)
                %                     nc = find(diff(id)~=1);
                %                     tor = [];
                %                     for j = 1:length(nc)
                %                         if id(nc(j)) + conThresh <= id(nc(j)+1)
                %                             tor = [tor;nc(j)];
                %                         end
                %                     end
                %                     id(tor) = [];
                %                 end
                
                switch noConsec
                    case 'true'
                        % new version
                        if ~isempty(id)
                            nc = find(diff(id)~=1);
                            rep = 1;
                            while rep == 1
                                tor = [];
                                for j = 1:length(nc)
                                    if id(nc(j)) + conThresh <= id(nc(j)+1)
                                        tor = [tor;nc(j)];
                                    end
                                end
                                id(tor) = [];
                                nc = find(diff(id)~=1);
                                if isempty(tor)
                                    rep = 0;
                                end
                            end
                        end
                end
                    
                
                switch autoSigmaAdj
                    case 'true'
                        sigmaTolNew(i,1) = sigmaTol;
                        while isempty(id)
                            sigmaTolNew(i,1) = sigmaTolNew(i,1)*autoAdjFac;
                            disp(['Sigma increased to : ' num2str(sigmaTolNew(i,1)) '; sub : ' num2str(i)])
                            n = sigmaTolNew(i,1)*sqrt(estimatenoise(d)); % 3 sigma tolerance; we take this approach because the plateau becomes centered around zero. This means we can do a simple test if we can estimate noise in the data (some sigma value)
                            pbool = (d <= n); % find values above "noise"
                            id = find(pbool == 1);
                            
                            % old version
                            %                             if ~isempty(id)
                            %                                 nc = find(diff(id)~=1);
                            %                                 tor = [];
                            %                                 for j = 1:length(nc)
                            %                                     if id(nc(j)) + conThresh <= id(nc(j)+1)
                            %                                         tor = [tor;nc(j)];
                            %                                     end
                            %                                 end
                            %                                 id(tor) = [];
                            %                             end
                            
                            switch noConsec
                                case 'true'
                                    % new version
                                    if ~isempty(id)
                                        nc = find(diff(id)~=1);
                                        rep = 1;
                                        while rep == 1
                                            tor = [];
                                            for j = 1:length(nc)
                                                if id(nc(j)) + conThresh <= id(nc(j)+1)
                                                    tor = [tor;nc(j)];
                                                end
                                            end
                                            id(tor) = [];
                                            nc = find(diff(id)~=1);
                                            if isempty(tor)
                                                rep = 0;
                                            end
                                        end
                                    end
                            end
                        end
                end
            case 'inflection'
                g = gradient(gradient(dTmp));
                id  = find(diff(sign(g)));
            case 'inflection noise'
                g = abs(gradient(gradient(dTmp)));
                n = sigmaTol*sqrt(estimatenoise(g));
                pbool = ((g) <= n);
                id = find(pbool == 1);
        end
        

        try
            dP(i,1) = id(1); % this is the edge of plateau
        catch % if id is still empty, sigma is still too low...so return k = 2
            disp(num2str(i))
            dP(i,1) = 1;
        end
        pos = 1:dP(i); % these are the potential solutions
        
        %% 2) Remove insignificant dip values from potential solutions
        id = find(dTmpP > pThresh); % these are solutions that do not pass threshold for unimodality
        [pos,~]= setdiff(pos,id); % these are the updated possible solution
        remPac = pTmp(pos);
        remPac = -1*remPac;
        pacs = std(remPac);
        remBC = bTmp(pos);
        bcs = std(remBC);
        
        % get global optima
        if ~isempty(remPac)
            [mVPac,globOptPac(i,1)] = max(remPac);
            globOptPac(i,1) = pos(globOptPac(i,1));
            [mVBC,globOptBC(i,1)] = max(remBC);
            globOptBC(i,1) = pos(globOptBC(i,1));
            
            % find peaks
            switch type
                case 'globMax'
                    if length(remPac) > 3
                        [pks,locs,~,~] = findpeaks(remPac); % now get peaks
                        pks2 = bTmp(pos(locs));
                        
                        if ~isempty(pks)
                            id = find(pks >= ((mVPac) - numStdDevs*pacs));
                            id2 = find(pks2 >= ((mVBC) - numStdDevs*bcs));
                            [id,~,~] = intersect(id,id2);
                            if ~isempty(id)
                                locOpt(i,1) = pos(locs(id(end)));
                            else
                                locOpt(i,1) = globOptPac(i,1);
                            end
                        else
                            id = find(remPac >= ((mVPac) - numStdDevs*pacs));
                            id2 = find(remBC >= ((mVBC) - numStdDevs*bcs));
                            [id,~,~] = intersect(id,id2);
                            if ~isempty(id)
                                locOpt(i,1) = pos(id(end));
                            else
                                locOpt(i,1) = globOptPac(i,1);
                            end
                        end
                    else
                        id = find(remPac >= ((mVPac) - numStdDevs*pacs));
                        id2 = find(remBC >= ((mVBC) - numStdDevs*bcs));
                        [id,~,~] = intersect(id,id2);
                        if ~isempty(id)
                            locOpt(i,1) = pos(id(end));
                        else
                            locOpt(i,1) = globOptPac(i,1);
                        end
                    end
                    
                case 'intersect'
                    if length(remPac) > 3
                        %% intersection
                        [pksPac,locsPac,~,~] = findpeaks(remPac); % now get peaks
                        [pksBC,locsBC,~,~] = findpeaks(remBC); % now get peaks
                        
                        % get intersection of BC and PAC peaks
                        [locs,ia,ib] = intersect(locsPac,locsBC);
                        pksPac = pksPac(ia);
                        pksBC = pksBC(ib);
                        if ~isempty(locs)
                            id = find(pksPac >= ((mVPac) - numStdDevs*pacs));
                            id2 = find(pksBC >= ((mVBC) - numStdDevs*bcs));
                            [id,~,~] = intersect(id,id2);
                            if ~isempty(id)
                                locOpt(i,1) = pos(locs(id(end)));
                            else
                                locOpt(i,1) = globOptPac(i,1);
                            end
                        else
                            id = find(remPac >= ((mVPac) - numStdDevs*pacs));
                            id2 = find(remBC >= ((mVBC) - numStdDevs*bcs));
                            [id,~,~] = intersect(id,id2);
                            if ~isempty(id)
                                locOpt(i,1) = pos(id(end));
                            else
                                locOpt(i,1) = globOptPac(i,1);
                            end
                        end
                    else
                        id = find(remPac >= ((mVPac) - numStdDevs*pacs));
                        id2 = find(remBC >= ((mVBC) - numStdDevs*bcs));
                        [id,~,~] = intersect(id,id2);
                        if ~isempty(id)
                            locOpt(i,1) = pos(id(end));
                        else
                            locOpt(i,1) = globOptPac(i,1);
                        end
                    end
            end
        else
            % this is what LH PPS was run with, but I think we should
            % assign best solution according to BC instead...
            locOpt(i,1) = 0;
            globOptPac(i,1) = 0;
            dP(i,1) = 0;
            
%             [~,locOpt(i,1)] = max(bTmp);
%             globOptPac(i,1) = locOpt(si,1);
%             dP(i,1) = locOpt(i,1);
            
        end
    else % if everything is zero
        locOpt(i,1) = 0;
        globOptPac(i,1) = 0;
        dP(i,1) = 0;
    end
end

for i = 1:length(locOpt)
    if locOpt(i) ~= 0
        locOpt(i,1) = kRange(locOpt(i));
        globOptPac(i,1) = kRange(globOptPac(i));
    else
        locOpt(i,1) = 0;
        globOptPac(i,1) = 0;
    end
end

%% this is the correlation bit at the end
% take the dip statistic and correlate bc and pac prior to plateau
for i = 1:size(p,2)
    % get current sub
    pTmp = p(:,i);
    dTmp = dip(:,i);
    bTmp = bc(:,i);
    
    if dP(i) ~= 0
        if length(dP(i)) > 2
            [postdPCorrR_pacbc(i,1),postdPCorrP_pacbc(i,1)] = corr(bTmp(dP(i):end),pTmp(dP(i):end));
            [postdPCorrR_pacdip(i,1),postdPCorrP_pacdip(i,1)] = corr(pTmp(dP(i):end),dTmp(dP(i):end));
            [predPCorrR_pacbc(i,1),predPCorrP_pacbc(i,1)] = corr(bTmp(1:dP(i)-1),pTmp(1:dP(i)-1));
            [predPCorrR_pacdip(i,1),poredPCorrP_pacdip(i,1)] = corr(pTmp(1:dP(i)-1),dTmp(1:dP(i)-1));
        else
            postdPCorrR_pacbc(i,1) = 0;
            postdPCorrP_pacbc(i,1) = 0;
            postdPCorrR_pacdip(i,1) = 0;
            postdPCorrP_pacdip(i,1) = 0;
            predPCorrR_pacbc(i,1) = 0;
            predPCorrP_pacbc(i,1) = 0;
            predPCorrR_pacdip(i,1) = 0;
            predPCorrP_pacdip(i,1) = 0;
        end
    else
        postdPCorrR_pacbc(i,1) = 0;
        postdPCorrP_pacbc(i,1) = 0;
        postdPCorrR_pacdip(i,1) = 0;
        postdPCorrP_pacdip(i,1) = 0;
        predPCorrR_pacbc(i,1) = 0;
        predPCorrP_pacbc(i,1) = 0;
        predPCorrR_pacdip(i,1) = 0;
        predPCorrP_pacdip(i,1) = 0;
    end
end

for i = 1:length(dP)
    if dP(i) ~= 0
        dP(i,1) = kRange(dP(i));
    else
        dP(i,1) = 0;
    end
end


