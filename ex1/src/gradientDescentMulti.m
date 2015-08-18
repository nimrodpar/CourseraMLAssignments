function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    h = X * theta;
    for j = 1:length(theta)
        theta(j) = theta(j) - alpha * (1/m) * sum(transpose(X(:,j))*(h - y));
    end
    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
