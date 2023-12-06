function [varargout] = etaSquared2(inData1,inData2,varargin)
% -------------------------------------------------------------------------
% [eta2] = etaSquared(inData1,inData2);
% [eta2,eta2Idx] = etaSquared(inData1,inData2,'Distance','true','Smallest',10);
% -------------------------------------------------------------------------
% etaSquared2 returns pairwise eta coefficients between observations in
% matrices inData1 and inData2. Input variables "inData1" and "inData2" are
% n1 x p1 and n2 x p2 matrices with possibly different sets of observations
% (n1 and n2) but the same set of variables (p1 and p2). Output "eta2"
% stores similarities between observations and is of size n1 x n2.
%
% Optional arguments: -----------------------------------------------------
% 'distance' -- when passed as 'true' (default is false), returns eta
% distances rather than coefficients (i.e., eta2 = 1 - eta2) so that lower
% numbers mean two observations are more similar.
%
% 'smallest' -- when passed and followed by some number k, eta2 will
% contain the k smallest pairwise distances from observations in inData2 to
% observations in inData1. The indices of these k smallest pairwise
% distances will be provided in eta2Idx.
%
% 'largest' -- does the same as 'smallest' but returns the k largest
% distances'
%
% More information about eta^2---------------------------------------------
% eta^2 is a symmetric similarity measure that can vary in value from 0 (no
% similarity) to 1 (identical) and presents an alternative to pearson's r
% when analyzing similarity between spatial maps. Very similar to
% correlation, but it's sensitive to magnitude at the level of individual
% pixels/observations/voxels. E.g. "if the value of each voxel in one map
% is 100 units greater than another, they will have a correlation
% coefficient of 1, even though every voxel is different.". 
%
% See "Defining Functional Areas in Individual Human Brains using Resting
% Functional Connectivity MRI" by Cohen et al (2008) in Neuroimage.
%
% Alex Teghipco // alex.teghipco@uci.edu

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

% Some sanity checks
if size(inData1,2) ~= size(inData2,2)
    error('The number of variables in your two input matrices does not match')
end
if ~isempty(options.smallest) && ~isempty(options.largest)
    error('You cannot select both k largest and k smallest distances at the same time')
end
if (~isempty(options.smallest) || ~isempty(options.largest)) && strcmp(options.distance,'false')
    options.distance = 'true'; % trigger distance if you are getting k largest or smallest components
    warning('You have selected either k largest or smallest observations. Eta coefficients will be changed to distances.')
end

% Main loops
for i = 1:size(inData1,1)
    %disp([num2str((i/size(inData1,1))*100) '%'])
    tmp1 = inData1(i,:);
    for j = 1:size(inData2,1)
        tmp2 = inData2(j,:);
        mbar = mean([tmp1 tmp2]);
        m = (tmp1 + tmp2)./2;
        num = ((tmp1 - m).^2) + ((tmp2 - m).^2);
        denom = ((tmp1 - mbar).^2) + ((tmp2 - mbar).^2);
        eta2(i,j) = (1 - sum(num)/sum(denom));
    end
end

switch options.distance
    case 'true'
        eta2 = 1 - eta2;
end

if ~isempty(options.smallest)
    for i = 1:size(eta2,2)
        [tmp, tmpIdx] = sort(eta2(:,i));
        eta2s(:,i) = tmp(1:options.smallest);
        eta2Idx(:,i) = tmpIdx(1:options.smallest);
    end
    eta2 = eta2s;
end

if ~isempty(options.largest)
    for i = 1:size(eta2,2)
        [tmp, tmpIdx] = sort(eta2(:,i),'descend');
        eta2s(:,i) = tmp(1:options.largest);
        eta2Idx(:,i) = tmpIdx(1:options.largest);
    end
    eta2 = eta2s;
end

varargout{1} = eta2;
if nargout == 2
    if ~isempty(options.smallest) || ~isempty(options.largest)
        varargout{2} = eta2Idx;
    else
        varargout{2} = 'You did not choose largest or smallest samples to return so there are no indices';
    end
end
