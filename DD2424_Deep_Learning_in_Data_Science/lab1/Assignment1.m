
%% Main
%%

% Parameters
K=10; %number of classes
d=3072; %number of dimensions (32*32*3)

% Read the data
[Xtrain,Ytrain,ytrain]=LoadBatch('data_batch_1.mat');
[Xtest,Ytest,ytest]=LoadBatch('data_batch_2.mat');

% Initialize W and b
rng(1);
W=0.01*randn(K,d);
b=0.01*randn(K,1);

% Mini-batch gradient descent
GDparams=[100;0.01;40];
lambda=0;
[Wstar,bstar] = MiniBatchGD(Xtrain,Ytrain,Xtest,Ytest,GDparams,W,b,lambda);

% Results
ComputeAccuracy(Xtest,Ytest,Wstar,bstar)

% Display the weights
for i=1:10
    im=reshape(Wstar(i,:),32,32,3);
    s_im{i}=(im-min(im(:)))/(max(im(:))-min(im(:)));
    s_im{i}=permute(s_im{i},[2,1,3]);
end
%montage(s_im)

%% check gradients
% K=10;
% d=3072;
% Xg=Xtrain(:,1:5);
% Yg=Ytrain(:,1:5);
% rng(1);
% W=0.01*randn(K,d);
% b=0.01*randn(K,1);
% P=EvaluateClassifier(Xg,W,b);
% lambda=1;
% [grad_W, grad_b ] = ComputeGradients(Xg,Yg,P,W,lambda);
% [grad_bnum, grad_Wnum] = ComputeGradsNumSlow(Xg, Yg, W, b, lambda, 1e-6);
%% Functions
%%
function [X,Y,y]=LoadBatch(filename)

A=load(filename);
N=length(A.data);

X=double(A.data)/255;
X=permute(X,[2,1,3,4]);

Y=zeros(10,N);
y=zeros(N,1);
for i=1:N
    Y(A.labels(i)+1,i)=1;
    y(i)=A.labels(i)+1;
end

end

%%
function P = EvaluateClassifier(X,W,b)

    N=size(X,2);

    s=W*X+b*ones(1,N);

    P=zeros(10,N);
    for i=1:N
        normalizationFactor=ones(1,10)*exp(s(:,i));
        for j=1:10
            P(j,i)=exp(s(j,i))/normalizationFactor;
        end
    end
        
end


%%
function acc = ComputeAccuracy(X,y,W,b)

    P=EvaluateClassifier(X,W,b);

    numberCorrectPredictions=0;
    for i=1:size(X,2)
        [~,argmax]=max(P(:,i));
        if y(argmax,i)==1 %the prediction is correct
            numberCorrectPredictions=numberCorrectPredictions+1;
        end
    end

    acc=numberCorrectPredictions/size(X,2);

end

%%
function J = ComputeCost(X,Y,W,b,lambda)

    P=EvaluateClassifier(X,W,b);

    crossEntropy=0;
    for i=1:size(X,2)
        crossEntropy=crossEntropy-log(Y(:,i)'*P(:,i));
    end

    sumWeights=0;
    for i=1:size(W,1)
        for j=1:size(W,2)
            sumWeights=sumWeights+power(W(i,j),2);
        end
    end

    J=crossEntropy/size(X,2)+lambda*sumWeights;

end

%%
function [grad_W, grad_b ] = ComputeGradients(X,Y,P,W,lambda)
 
    numberData=size(X,2);
    numberDim=size(X,1);
    
    grad_b=zeros(10,1);
    grad_W=zeros(10,numberDim);
    
    g=-(Y-P);
    grad_b=sum(g,2);
    grad_W=g*X';
%     for i=1:numberData
%         g=-(Y(:,i)-P(:,i));
%         grad_b=grad_b+g;
%         grad_W=grad_W+g*X(:,i)';
%     end
    grad_b=grad_b/numberData;
    grad_W=grad_W/numberData+2*lambda*W;

end

%%
function [Wstar,bstar] = MiniBatchGD(Xtrain,Ytrain,Xtest,Ytest,GDparams,W,b,lambda)

    n_batch=GDparams(1);
    eta=GDparams(2);
    n_epochs=GDparams(3);
    N=size(Xtrain,2);
    
    hold on
    for t=1:n_epochs
        %disp(t)
        disp(ComputeAccuracy(Xtest,Ytest,W,b))
          plot(t,ComputeCost(Xtrain,Ytrain,W,b,lambda),'r.') %print the cost on the training set
          plot(t,ComputeCost(Xtest,Ytest,W,b,lambda),'b.') %print the cost on the test set
          legend('training cost','test cost')
          xlabel('n epochs')
        % compute the loss
        sumWeights=0;
        for i=1:size(W,1)
            for j=1:size(W,2)
                sumWeights=sumWeights+power(W(i,j),2);
            end
        end
%         plot(t,ComputeCost(Xtrain,Ytrain,W,b,lambda)-lambda*sumWeights,'r.') %print the loss on the training set
%         plot(t,ComputeCost(Xtest,Ytest,W,b,lambda)-lambda*sumWeights,'b.') %print the loss on the test set
%         legend('training loss','test loss')
%         xlabel('n epochs')
        for j=1:N/n_batch
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            Xbatch = Xtrain(:, j_start:j_end);
            Ybatch = Ytrain(:, j_start:j_end);
            Pbatch=EvaluateClassifier(Xbatch,W,b);
            [gradW,gradb]=ComputeGradients(Xbatch,Ybatch,Pbatch,W,lambda);
            W=W-eta*gradW;
            b=b-eta*gradb;
        end
    end
    Wstar=W;
    bstar=b;
end
%%
function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)

no = size(W, 1);
d = size(X, 1);

grad_W = zeros(size(W));
grad_b = zeros(no, 1);

c = ComputeCost(X, Y, W, b, lambda);

for i=1:length(b)
    b_try = b;
    b_try(i) = b_try(i) + h;
    c2 = ComputeCost(X, Y, W, b_try, lambda);
    grad_b(i) = (c2-c) / h;
end

for i=1:numel(W)   
    
    W_try = W;
    W_try(i) = W_try(i) + h;
    c2 = ComputeCost(X, Y, W_try, b, lambda);
    
    grad_W(i) = (c2-c) / h;
end
end
%%
function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

no = size(W, 1);
d = size(X, 1);

grad_W = zeros(size(W));
grad_b = zeros(no, 1);

for i=1:length(b)
    b_try = b;
    b_try(i) = b_try(i) - h;
    c1 = ComputeCost(X, Y, W, b_try, lambda);
    b_try = b;
    b_try(i) = b_try(i) + h;
    c2 = ComputeCost(X, Y, W, b_try, lambda);
    grad_b(i) = (c2-c1) / (2*h);
end

for i=1:numel(W)
    
    W_try = W;
    W_try(i) = W_try(i) - h;
    c1 = ComputeCost(X, Y, W_try, b, lambda);
    
    W_try = W;
    W_try(i) = W_try(i) + h;
    c2 = ComputeCost(X, Y, W_try, b, lambda);
    
    grad_W(i) = (c2-c1) / (2*h);
end

end