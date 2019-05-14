function [d] = onehotenc(nclasses, k)
%ONEHOTENC return a onehot vector of class k
%   vector with zeros and 1 at position k
    
    y_one_hot = zeros(nclasses, 1);
    for i = 1:nclasses
        y_one_hot(k, 1) = 1
    
    d = y_one_hot
end
