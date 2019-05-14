%% Assessment 2
clear
rng('default');

%% loading data
load mnist.mat

% rename label 0 to 10
train_labels(train_labels == 0) = 10;
test_labels(test_labels == 0)   = 10;
labels = unique(train_labels);

%% Neural Network

% Fixed parameters
d = size(train_data, 2); % MNIST digit size 
nclasses = length(labels); % total number of classes
Ni = d; % Number of external inputs
Nh = 200; % Number of hidden units
No = nclasses; % Number of output units
alpha_i = 0.0; % Input weight decay
alpha_o = 0.0; % Output weight decay
range = 0.1; % Initial weight range                
eta=0.001; % gradient descent parameter

% Initialize network weights
Wi = range * randn(Nh,Ni+1);
Wo = range * randn(No,Nh+1);

max_iter=5;             % maximum number of iterations
iter = 1;
fprintf('Training ...\n');
while iter < max_iter
  fprintf('Iteration %d ...\n', iter);
  % implement gradient descent updates here
  % hint use fullGradient
  [dWi, dWo] = fullGradient(Wi,Wo,alpha_i,alpha_o, train_data, train_labels, nclasses);
  Wi = Wi - eta*dWi;
  Wo = Wo - eta*dWo;
  
  
  
  iter = iter + 1;
end



% Test and print accuracy
fprintf('Testing ...\n');
acc = 0;
N   = length(test_labels);N=5;
y_pred = zeros(size(test_labels));
for k = 1:N
  yi = [1;train_data(k, :)']; % input
  v1 = Wi*yi; % FC
  yh = [1;relu(v1)]; % hidden layer w/ bias
  
  % output layer
  v2 = Wo*yh; % FC
  yo = softmax(v2); % softmax
  
  [~, i] = max(yo);
  y_pred(k) = i;
  if i == train_labels(k)
    acc = acc + 1;
  end
  if(rem(k, 100)==0)
    fprintf('%d done.\n', k);
  end
end

acc = acc / N;
fprintf('Accuracy is %f\n', acc);
