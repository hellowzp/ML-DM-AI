function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%	GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); 	% number of training examples
J_history = zeros(num_iters, 1);

for iter = 1 : num_iters
    J_derivative_1 = 0;
    J_derivative_2 = 0;
    for  i = 1 : m
        h = transpose(theta) * transpose( X(i,:));
        J_derivative_1 = J_derivative_1 + 1/m * (h - y(i)) * X(i,1);
        J_derivative_2 = J_derivative_2 + 1/m * (h - y(i)) * X(i,2);
    end
    theta(1,1) = theta(1,1) - alpha * J_derivative_1;
    theta(2,1) = theta(2,1) - alpha * J_derivative_2;
    
    J_history(iter) = computeCost(X, y, theta);
end

% for iter = 1 : num_iters
%     for j = 1 : size(theta)
%         J_derivative = 0;
%         for  i = 1 : m
% 			h = transpose(theta) * transpose( X(i,:));
%             J_derivative = J_derivative + 1/m * (h - y(i)) * X(i,j);
%         end
%         theta(j,1) = theta(j,1) - alpha * J_derivative;
%     end
%     J_history(iter) = computeCost(X, y, theta);
% end

% for iter = 1 : num_iters
% %	printf("iteration %d:\n", iter);
% 	% ====================== YOUR CODE HERE ======================
% 	% Instructions: Perform a single gradient step on the parameter vector
% 	%               theta for each iteration. 
% 	%
% 	% Hint: While debugging, it can be useful to print out the values
% 	%       of the cost function (computeCost) and gradient here.
% 	for j = 1 : size(theta)
% 		J_derivative = 0;
% 		for i = 1 : m
% 			h = transpose(theta) * transpose( X(i,:));
% 			J_derivative += 1/m * (h - y(i)) * X(i,j);
%         end
% 		
% 		% mind that although theta is a column vector, operations about it
% 		% is still treated as matrix operation
% 		theta(j,1) = theta(j,1) - alpha * J_derivative; 
% 		
% %		printf("theta-%d: %f\n", j-1, theta(j,1));
%     end
% 
%     % ============================================================
% 
%     % Save the cost J in every iteration    
%     J_history(iter) = computeCost(X, y, theta);
% 	
% %	printf("J: %f\n\n", J_history(iter));
% 
% end

end
