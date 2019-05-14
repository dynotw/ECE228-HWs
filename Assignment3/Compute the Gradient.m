function [dWi,dWo] = fullGradient(Wi,Wo,alpha_i,alpha_o, Inputs, Labels, nclasses)
%
% Calculate the partial derivatives of the quadratic cost function
% wrt. the weights. Derivatives of quadratic weight decay are included.
%
% Input:
%        Wi      :  Matrix with input-to-hidden weights
%        Wo      :  Matrix with hidden-to-outputs weights
%        alpha_i :  Weight decay parameter for input weights
%        alpha_o :  Weight decay parameter for output weights
%        Inputs  :  Matrix with examples as rows
%        Targets :  Matrix with target values as rows
% Output:
%        dWi     :  Matrix with gradient for input weights
%        dWo     :  Matrix with gradient for output weights

% Determine the number of examples
[ndata, inp_dim] = size(Inputs);

dWi = zeros(size(Wi));
dWo = zeros(size(Wo));

% compute the derivatives for each data point separately.
for k=1:ndata
  %%%%%%%%%%%%%%%%%%%%
  %%% FORWARD PASS %%%
  %%%%%%%%%%%%%%%%%%%%
  %
  % Propagate kth example forward through network
  % calculating all hidden- and output unit outputs
  %
  
  % Calculate hidden unit outputs for every example
  x = cat(2,[1],Inputs(k,:))';    % 785*1   
  g = Wi*x;    % 200*1
  
  % ReLU
  h = cat(1,[1],relu(g));   % 201*1
  y = softmax(Wo*h);     % 10*1

  
  %%%%%%%%%%%%%%%%%%%%%
  %%% BACKWARD PASS %%%
  %%%%%%%%%%%%%%%%%%%%%
  % Back propagation
  % Calculate derivative of
  % by backpropagating the errors from the
  % desired outputs

  t = onehotenc(nclasses, Labels(k))
  Do = y - t;
  Di = (g>0) .* (Wo(:,2:201)'* Do);
  dWo = dWo + Do * h';
  dWi = dWi + Di * x';
  
end


% Add derivatives of the weight decay term

dWo = dWo/ndata + alpha_o*Wo;
dWi = dWi/ndata + alpha_i*Wi;

end



