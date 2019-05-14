N       = 10000;                                        %number of samples
p       = randn(N,1);                                   %Gaussian random samples
bins    = -10:.1:10;                                    %histogram bins
hp      = hist(p,bins);                                 %values in histogram bins
P       = normpdf(bins);                                %true distribution
p2      = randn(N,1);                                   %more Gaussian random samples
hp2     = hist(p2,bins);                                %more values in histogram bins

% generate laplacian random variables and their pdf (mean zero beta = 1)
beta    = 1;                                            %laplacian RV parameter
q       = rand(N,1)-.5;                                 %generate laplacian RV according to wiki
q       = real(sign(q).*log(1-2*abs(q)));               %thank god for wiki
Q       = ( 1/(2*beta) ) * exp(-abs(bins)/beta);        %keep thanking
hq      = hist(q,bins);                                 %estimated distribution

KL_PQ   = KLdiv(P,Q);                                   %KL(P || Q)
KL_QP   = KLdiv(Q,P);                                   %KL(Q || P)
KL_est  = KLdiv(hp,hq);                                 %KL of the estimated distributions


%% Plot figure 1
txt1 = sprintf('KL(P || Q) = %f', KL_PQ);
txt2 = sprintf('KL(Q || P) = %f', KL_QP);
figure(1)
subplot(2,1,1)
hist(p,bins)
hold on
plot(bins, P*N/9.9)
axis([-5,5,0,600])
grid on
title('N = 10,000 samples from distribution P (Standard Normal)')
text(2,500,txt1)


subplot(2,1,2)
hist(q,bins)
hold on
plot(bins, Q*N/9.9)
axis([-5,5, 0,600])
grid on
title('N = 10,000 samples from distribution Q (Laplacian, \beta = 1)')
text(2,500,txt2)

%% Plot figure 2

KL_PQ   = KLdiv(hp,hp2);                                   %KL(P || Q)
KL_QP   = KLdiv(hp2,hp);                                   %KL(Q || P)

assert(KL_PQ < .01, 'KL divergence between randomly generated Gaussian distributions above realistic threshhold');
assert(KL_QP < .01, 'KL divergence between randomly generated Gaussian distributions above realistic threshhold');

txt3 = sprintf('KL(P || Q) = %f', KL_PQ);
txt4 = sprintf('KL(Q || P) = %f', KL_QP);
figure(2)
subplot(2,1,1)
hist(p,bins)
hold on
plot(bins, P*N/9.9)
axis([-5,5,0,600])
grid on
title('N = 10,000 samples from distribution P (Standard Normal)')
text(2,500,txt3)


subplot(2,1,2)
hist(p2,bins)
hold on
plot(bins, P*N/9.9)
axis([-5,5, 0,600])
grid on
title('N = 10,000 samples from distribution Q (Standard Normal)')
text(2,500,txt4)

