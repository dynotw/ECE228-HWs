function [y] = softmax(z)
% z is a K x 1 vector of floats.
% y should be the output of softmax function
    dim = 1;
    s = ones(1, ndims(z));
    s(dim) = size(z, dim);
    maxz = max(z, [], dim);
    expz = exp(z-repmat(maxz, s));
    y = expz ./ repmat(sum(expz, dim), s);

end
