function [BC] = bimodalCoeff(inData)
% function [BC] = bimodalCoeff(inData)
% Returns bimodal coefficient for inData.
% alex.teghipco@uci.edu

m3 = skewness(inData, 0);
m4 = kurtosis(inData, 0) - 3;
n = (m3.^2) + 1;
d1 = ((length(inData)-1).^2 / ((length(inData) - 2)*(length(inData) - 3)));
d2 = (m4+3)*d1;
BC = n/d2;
