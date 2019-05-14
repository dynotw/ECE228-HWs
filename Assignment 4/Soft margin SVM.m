function [a, b] = softsvm(X, t, C)
    K = X*X'.*(t*t');
    e = ones(size(X,1),1);
    options = optimoptions('quadprog');
    options.MaxIterations = 9000;
    LB = repmat(0,size(X,1),1);
    UB = repmat(C,size(X,1),1);
    a = quadprog(K,e,[],[],t',0,LB,UB,[],options);
    b = (t-X*X'*(a.*t))/size(X,1);
end
