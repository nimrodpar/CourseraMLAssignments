function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

a1 = [ones(m, 1) X];
z2 = (Theta1*a1')';
a2 = [ones(m, 1) sigmoid(z2)];
z3 = (Theta2*a2')';
a3 = sigmoid(z3);
h = a3;

for i=1:m
    y_vec = zeros(num_labels,1);
    y_vec(y(i)) = 1;
    for k=1:num_labels
        J = J + -y_vec(k)*log(h(i,k)) - (1-y_vec(k))*log(1-h(i,k));
    end
    
end 
J = J / m;

R = 0;
for j=1:hidden_layer_size
    for k=2:input_layer_size+1 % we do not regularize the bias..
        R = R + Theta1(j,k)^2;
    end
end

for j=1:num_labels
    for k=2:hidden_layer_size+1
        R = R + Theta2(j,k)^2;
    end
end
J = J + lambda * R / (2*m);

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

Delta1 = zeros(size(Theta1_grad));
Delta2 = zeros(size(Theta2_grad));
for t = 1:m
	a1_t = [1 ; X(t,:)'];
	z2_t = Theta1 * a1_t;
	a2_t = [1 ; sigmoid(z2_t)];
	z3_t = Theta2 * a2_t;
	a3_t = sigmoid(z3_t);

	delta3 = a3_t - (y(t) == [1:num_labels]');
	Delta2 = Delta2 + delta3 * a2_t';

	delta2 = (Theta2' * delta3)(2:end) .* sigmoidGradient(z2_t);
	Delta1 = Delta1 + delta2 * a1_t'; 
end

Theta1_grad = Delta1 ./ m;
Theta2_grad = Delta2 ./ m;
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

for i = 1:size(Delta1)(1)
	for j = 2:size(Delta1)(2)
		Theta1_grad(i,j) = Theta1_grad(i,j) + Theta1(i,j) * lambda / m;
	end
end

for i = 1:size(Delta2)(1)
	for j = 2:size(Delta2)(2)
		Theta2_grad(i,j) = Theta2_grad(i,j) + Theta2(i,j) * lambda / m;
	end
end



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
