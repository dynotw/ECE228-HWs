function [y] = relu(x)
%RELU implements the relu activation function.
    new_x =max(0,x);
    y = new_x
end
