function [J, grad] = costFunctionReg(theta, X, y, lambda)
%   COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% % optimized algorithm
% for i = 1 : m
%     h = sigmoid( X(i,:) * theta );
%     J = J -  y(i) * log(h) - ( 1- y(i)) * log(1-h);
% end
% J = J/m;
% 
% % parameter theta-0 is not regularized, calculate separately
% grad(1,:) = 1/m * ( sigmoid( X(i,:) * theta ) - y(i)) * X(i,1);
% 
% % regularized part of cost and other gradients
% J_Reg = 0;
% for j = 2 : size(theta, 1)
%     J_Reg = J_Reg + power( theta(j), 2);
%     
%     % iterate over every other parameter theta for other gradients
%     for i = 1 : m
%         h = sigmoid( X(i,:) * theta );
%         grad(j,:) = grad(j,:) + ( h - y(i)) * X(i,j);
%     end
%     grad(j,:) = 1/m * grad(j,:) + lambda/m * theta(j,:);
% end
% J = J + lambda * 1/(2*m) * J_Reg;

for i = 1 : m
    h = sigmoid( X(i,:) * theta );
    J = J -  y(i) * log(h) - ( 1- y(i)) * log(1-h);
end
J = J/m;

% regularized part of cost and gradients
J_Reg = 0;
%[rows, cols] = size(theta);  % or rows = size(theta, 1);
for j = 2 : size(theta, 1)
    J_Reg = J_Reg + power( theta(j), 2);
end
J = J + lambda * 1/(2*m) * J_Reg;

% iterate over every parameter theta to calculate the gradients
for j = 1 : size(theta, 1)
    for i = 1 : m
        h = sigmoid( X(i,:) * theta );
        grad(j,:) = grad(j,:) + ( h - y(i)) * X(i,j);
    end
    grad(j,:) = 1/m * grad(j,:) + lambda/m * theta(j,:);  
end   

% parameter theta-0 is not regularized, calculate separately
grad(1,:) = grad(1,:) - lambda/m * theta(1,:);

% =============================================================

end
