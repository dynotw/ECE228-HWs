function [x] = OMPnorm(A,y,K)
% Orthogonal Matching Pursuit (OMP)
% Solve y = Ax, assuming x is a sparse vector with sparsity numOMP.
% [x] = OMPnorm(A,y,numOMP)
% [INPUTS]
% A - dictionary (N x M)
% y - Data vector of size M x 1
% K - desired sparsity (number of atoms)
% [OUTPUTS]
% x - Sparse coefficient matrix of size M x 1

[N, M] = size(A);

if (N ~= size(y))
    error('Dimension not Correct! size(y) != size(A)(1)');
end

x = zeros(M,1);
residual = y;                                            % initial residual is full measurement vector
indx = zeros(K,1);
A_cols = [];

for j = 1:K
    x_tmp = zeros(M,1);
    inds = setdiff([1:M],indx);                          % iterate all columns except for the chosen ones
    for i = inds
        x_tmp(i) = A(:,i)' * residual / norm(A(:,i));    % solve min ||a'x-b||
    end
    
    [~,pos] = max(abs(x_tmp));                           % find index number of maximum value of proj
    
    indx(j) = pos;                                       % store indices
    A_cols = [A_cols A(:,pos)];
    x_ls = A_cols \ y;                                   % Obtain Least-Square x
    residual = y - A_cols * x_ls;                        % update residual
end

for i = 1:K
    x(indx(i)) = x_ls(i);                                % x_sparse(i).value;
end
