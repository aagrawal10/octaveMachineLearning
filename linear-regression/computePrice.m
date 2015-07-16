function price = computePrice(x, theta)
%COMPUTEPRICE Compute price for passed feature vector and calculated value of theta
%   price = COMPUTEPRICE(x, theta) 

% The main vector to compute price
price = theta' * x;

end
