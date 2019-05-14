function KLD=KLdiv(P,Q)
    
    % normalizing the P and Q
    Q = Q ./ sum(Q);
    P = P ./ sum(P);
    KLD_P = safe_sum(P .*log(P ./Q));
    KLD_Q = safe_sum(Q .*log(Q ./P));

    KLD = -(KLD_P + KLD_Q) / 2;


    function d = safe_sum(d)
        d(isnan(d))=0; % resolving the case when P(i)==0
        d(isinf(d))=0; % resolving the case when Q(i)==0
        d = sum(d);
    end
end
