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



a_1=[ones(m,1) X];
temp=sigmoid(a_1*Theta1')
a_2=[ones(m,1) temp];
a_3=sigmoid(a_2*Theta2');

y=repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels);
J=(-1 / m) * sum(sum(y.*log(a_3) + (1 - y).*log(1 - a_3)));

reg_Theta1=Theta1(:,2:end);
reg_Theta2=Theta2(:,2:end);



err=((0.5*lambda)/m)*(sum(sum(reg_Theta1.^2))+sum(sum(reg_Theta2.^2)));

J=J+err;

temp1= zeros(size(Theta1));
temp2= zeros(size(Theta2));

for t = 1:m,
	t_a_1=a_1(t,:);
	t_a_2=a_2(t,:);
	t_a_3=a_3(t,:);

	t_y= y(t,:);

	l3=t_a_3-t_y;
	l2=Theta2'*l3' .* sigmoidGradient([1;Theta1 * t_a_1']);

	temp1=temp1+l2(2:end)*t_a_1;
	temp2=temp2+l3'*t_a_2;
end;

Theta1_grad = 1/m * temp1 + (lambda/m)*[zeros(size(Theta1, 1), 1) reg_Theta1];
Theta2_grad = 1/m * temp2 + (lambda/m)*[zeros(size(Theta2, 1), 1) reg_Theta2];



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
