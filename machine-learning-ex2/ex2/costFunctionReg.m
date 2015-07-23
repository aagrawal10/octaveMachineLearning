function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

J = -1.0 / m * ((y' * log(sigmoid(X * theta))) + ((1-y)' * log(1 - sigmoid(X * theta))));
J = J + 1.0 / (2 * m) * lambda * theta(2:size(theta))' * theta(2:size(theta));
grad = 1.0 / m * X' * (sigmoid(X * theta) - y) + lambda / m * theta;
grad(1) = grad(1) - lambda / m * theta(1);

% =============================================================

end
