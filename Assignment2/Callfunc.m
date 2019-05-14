
mu              = 0.0;                           % true mean value
sigma           = 1.0;                           % true variance
xmin            = -5.0;                          % min x value
xmax            = 5.0;                           % max x value
Npdf            = 100;                           % number of points in density
dx              = 0.4;                           % bin width for histogram
[x1,p1,x2,p2]   = norm1d(mu,sigma,xmin,xmax,Npdf,dx);

%% Plot 1
figure(1), subplot(2,1,1), plot(x1,p1,'b'), title('density of the 1D normal dist')
figure(1), subplot(2,1,2), bar(x2,p2,'b'), title('histogram of the 1D normal dist')

% generate normal variates using MATLAB randn function
M               = 100;                           % number of samples
y1              = sqrt(sigma)*randn(M,1)+mu*ones(M,1);
y2              = hist(y1,x2);
y2              = y2/sum(y2);

%% Plot 2
figure(2), subplot(2,1,1),  bar(x2,y2,'r'),hold on, plot(x2,p2,':'),hold off, title('histogram of samples and "true" histogram')
figure(2), subplot(2,1,2),  bar(x2,p2,'b'), title('"true" histogram')
