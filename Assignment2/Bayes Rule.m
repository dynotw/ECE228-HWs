xmin            = -10;                           % minimum value
xmax            = 10;                            % maximum value
Npdf            = 1000;                          % number of points in grid   
dx              = 0.4;                           % bin width

% Prior probabilities of 3 separate probability distributions
P1              = 0.4;
P2              = 0.3;
P3              = 1-P1-P2;

% Definitions of three normal distributions. Refer to these distributions as P1, P2, and P3

mu_x1           = -2;
sig_x1          = 1;
[x1,p11,x2,p12] = norm1d(mu_x1,sig_x1,xmin,xmax,Npdf,dx);
mu_x2           = 0;
sig_x2          = 1;
[x1,p21,x2,p22] = norm1d(mu_x2,sig_x2,xmin,xmax,Npdf,dx);
mu_x3           = 2;
sig_x3          = 1;
[x1,p31,x2,p32] = norm1d(mu_x3,sig_x3,xmin,xmax,Npdf,dx);

figure(3),subplot(3,1,1), plot(x1,p11,'b',x1,p21,'r',x1,p31,'g'), title('class conditional densities p(x|c)')

%% Find the full class conditional univariate normal distribution

px              = p11*P1+p21*P2+p31*P3; %vector containing combined probability distribution of P1, P2, and P3

% Use Bayes Rule to compute the posterior distribution of P1, P2, and P3

P1X             = (p11*P1)./px; %
P2X             = (p21*P2)./px; %
P3X             = (p31*P3)./px; %

figure(3), subplot(3,1,2), plot(x1,px,'m'), title('density p(x)')
figure(3), subplot(3,1,3), plot(x1,P1X,'b',x1,P2X,'r',x1,P3X,'g'),title('posterior class probabilities P(c|x)')
