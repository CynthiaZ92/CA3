function back_gradient_descent(w3, W2, W1, X_train, y_train, eta)
[m1,n1] = size(X_train);
% [m2,n2] = size(y_train);
%-----------process all the inputs------------
        for num = 1:m1  
        x = X_train(num,:);       
        y = y_train(num,:);       
        net2 = W1 * x.';              
        for i=1:n1
             hidden1(i)=1/(1+exp(-net2(i)));
        end
        net3 = W2 * hidden1.';       
        for i=1:n1
             hidden2(i)=1/(1+exp(-net3(i)));
        end
        o = w3.' * hidden2.';
%%-------------BP algorithm-----------------
       %start from last layer
        delta3 = (y-o)*o*(1-o);
       %second hidden layer
        for j = 1:n1     
             delta2(j) = hidden2(j)*(1-hidden2(j))*w3(j,:)*delta3;
        end
       %first hidden layer
        for k = 1:n1
             delta1(k) = hidden1(k)*(1-hidden1(k))*W2(k,:)*delta2.';
        end
%--------update W1,2,3---------------------
        for i = 1:n1 %w = w + eta*delta*x      
            w3(i,1) = w3(i,1) - eta*delta3*hidden2(i);
        end
        for i = 1:n1
                for j = 1:n1
                     W2(i,j) = W2(i,j) - eta*delta2(i)*hidden1(j);
                end
        end
        for i = 1:n1
                for j = 1:n1
                     W1(i,j) = W1(i,j) - eta*delta1(i)*x(j);
                end
        end
%--------------error
            e=o-y;%error
            sigma(num)=e*e;
            plot(sigma);
        end
end