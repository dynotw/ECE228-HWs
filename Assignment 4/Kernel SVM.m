function [a, b] = softsvm_RBF(X, t, C)
  nms = sum(X'.^2,1);
  n = size(X,1);
  K = exp(-(nms'*ones(1,n) + ones(n,1)*nms - 2*X*X'));
  e = ones(size(X,1),1);
  LB = repmat(0,size(X,1),1);
  UB = repmat(C,size(X,1),1);
  a = quadprog(K,e,[],[],t',0,LB,UB,[]);
  b = (t-X*X'*(a.*t))/size(X,1);
end
