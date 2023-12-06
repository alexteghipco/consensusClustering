function [pacOptAdj,optAdj] = adjustOptCon(optSol,pac,nstd,kRange)

% get pac for locally optimal solutions
tor = [];
for i = 1:size(pac,2)
    if optSol(i) ~= 0
        optAdj(i,1) = find(kRange == optSol(i));
        pacOpt(i,1) = pac(optAdj(i),i);
    else
        pacOpt(i,1) = 0;
        optAdj(i,1) = 0;
        tor = [tor; i];
    end
end

% remove subjects with empty cols
subs = 1:size(pac,2);
pacOptB = pacOpt;
subs(tor) = [];
pacOptB(tor) = [];
m = mean(pacOptB);
s = std(pacOptB);
b = m - (nstd*s);
a = m + (nstd*s);

% get outliers
o = (pacOptB < b) + (pacOptB > a);
subsAdj = subs(find(o == 1));

% loop over subjects with outlying pac
pacOptAdj = pacOpt;
for i = 1:length(subsAdj)
    tmp = pac(:,subsAdj(i));
    oId = find(tmp == pacOpt(subsAdj(i)));
    % if the subject had below-average pac (good pac), move k to
    % wherever pac would be closest to the mean
    if pacOpt(subsAdj(i)) <= b
        % move k up to first pac that hits above mean
        id1 = find(tmp > m); 
        id2 = find(id1 > oId);
        if ~isempty(id2)
            optAdj(subsAdj(i)) = id1(id2(1));
            pacOptAdj(subsAdj(i)) = tmp(id1(id2(1)));
        else
            optAdj(subsAdj(i)) = 0;
            pacOptAdj(subsAdj(i)) = 0;
        end
    % if the subject had above-average pac (poor pac), move k down to
    % increase pac within mean
    elseif pacOpt(subsAdj(i)) >= a
        % move k down to first pac that hits above mean
        id1 = find(tmp < a);
        id2 = find(id1 < oId);
        if ~isempty(id2)
            optAdj(subsAdj(i)) = id1(id2(1));
            pacOptAdj(subsAdj(i)) = tmp(id1(id2(1)));
        else
            optAdj(subsAdj(i)) = 0;
            pacOptAdj(subsAdj(i)) = 0;
        end
    end
end

% now adjust pac opt for kRange
for i = 1:length(optAdj)
    if optAdj(i) ~= 0
        optAdj(i,1) = kRange(optAdj(i));
    else
        optAdj(i,1) = 0;
    end
end

