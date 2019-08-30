
%% Read the data
data_fname = 'ascii_names.txt';

fid = fopen(data_fname,'r');
S = fscanf(fid,'%c');
fclose(fid);
names = strsplit(S, '\n');
if length(names{end}) < 1        
    names(end) = [];
end
ys = zeros(length(names), 1);
all_names = cell(1, length(names));
for i=1:length(names)
    nn = strsplit(names{i}, ' ');
    l = str2num(nn{end});
    if length(nn) > 2
        name = strjoin(nn(1:end-1));
    else
        name = nn{1};
    end
    ys(i) = l;
    all_names{i} = name;
end

C = unique(cell2mat(all_names));
% d = numel(C);
d=55;
K=length(unique(ys));

%% Encode the names

% Find n_len
n_len=0;
for i=1:length(all_names)
    n_len=max(n_len,length(all_names{i}));
end

% Create the map
keys=split(C,'');
keys=keys(2:end-1);
val=(1:length(keys));
map=containers.Map(keys,val);

% Create the matrix
N=length(all_names);
mat_names=zeros(N,d,n_len);

Ys=zeros(N,K);

for i=1:N
    for j=1:length(all_names{i})
        mat_names(i,map(all_names{i}(j)),j)=1;
    end
    Ys(i,ys(i))=1;
end

X=reshape(mat_names,[20050,d*n_len])';
 
%% Initialize the network
n1=20;
k1=5;
n2=20;
k2=3;
fsize=n2;
ConvNet.F{1}=randn(d,k1,n1)*0.32;%sqrt(2/(n_len*d*k1));
ConvNet.F{2}=randn(n1,k2,n2)*0.08;%sqrt(2/(n1*k2));
ConvNet.W=randn(K,((n_len-k1+1)-k2+1)*n2)*0.09;%*sqrt(2/(n2*((n_len-k1+1)-k2+1)*n2));

GDparams(1)=90; %n_batch
GDparams(2)=0.01; %eta
GDparams(3)=3000; %epochs
GDparams(4)=0; %rho

%% Check the gradient

% sample
x=X(:,1:10);
y=Ys(1:10,:)';

% compute the gradients
MX1=buildMX1(x,d,k1,n1);
[gradW,gradF1,gradF2] = ComputeGradients(x,y,ConvNet,MX1);
gradF1=reshape(gradF1,size(ConvNet.F{1}));
Gs = NumericalGradient(x,y,ConvNet,1e-5);

% reshape the gradients
gradF2=reshape(gradF2,size(ConvNet.F{2}));
gradW=reshape(gradW,size(ConvNet.W));

% compute the error
disp("max relative error for each coefficient of the gradient");
disp("wrt F1"); disp(max(max(max(abs(Gs{1}-gradF1)./max(1e-5,abs(Gs{1})+abs(gradF1))))));
disp("wrt F2"); disp(max(max(max(abs(Gs{2}-gradF2)./max(1e-5,abs(Gs{2})+abs(gradF2))))));
disp("wrt W"); disp(max(max(max(abs(Gs{3}-gradW)./max(1e-5,abs(Gs{3})+abs(gradW))))));

%%
d=55;

n1=20;
k1=5;
n2=20;
k2=3;

ConvNet.F{1}=randn(d,k1,n1)*sqrt(2/n_len);
ConvNet.F{2}=randn(n1,k2,n2)*sqrt(2/(n1*(n_len-k1+1)));
ConvNet.W=randn(K,((n_len-k1+1)-k2+1)*n2)*sqrt(2/(n2*((n_len-k1+1)-k2+1)*n2));

% [Xtrain,Ytrain,Xtest,Ytest]=createSample(X,Ys);
% Xtrain=sparse(Xtrain);
% Xtest=sparse(Xtest);
% Ytrain=sparse(Ytrain);
% Ytest=sparse(Ytest);

GDparams(1)=90; %n_batch
GDparams(2)=0.001; %eta
GDparams(3)=25000; %epochs
GDparams(4)=0.9; %rho


%MX1_20 = buildMX1(Xtrain,d,k1,n1);

[Convstar,res] = MiniBatchGD(Xtrain,Ytrain,Xtest,Ytest,GDparams,ConvNet,MX1_20);

%%
MFs = buildMFs(best,19);
[pu,~,~] = EvaluateClassifier(Xtest,MFs,best.W);
[~,Ytrue]=max(Ytest);
[~,Ypredictu]=max(pu);
mu=confusionmat(Ytrue,Ypredictu);
disp(mu);
disp(mu./sum(mu,2));
% imshow(mu./sum(mu,2), [], 'InitialMagnification', 1600);
% colorbar;
% axis on;


%% create the training and validation set
function [Xtrain,Ytrain,Xtest,Ytest]=createSample(X,Y)
    
    [d,N]=size(X);
    [~,K]=size(Y);
    newXtrain=zeros(d,0.8*N);
    newYtrain=zeros(0.8*N,K);
    newXtest=zeros(d,0.2*N);
    newYtest=zeros(0.2*N,K);
    
    ktrain=1;
    ktest=1;
    for i=1:N
       if mod(i,10)>1
           newXtrain(:,ktrain)=X(:,i);
           newYtrain(ktrain,:)=Y(i,:);
           ktrain=ktrain+1;
       else
           newXtest(:,ktest)=X(:,i);
           newYtest(ktest,:)=Y(i,:);
           ktest=ktest+1;
       end
    end
    
     Xtrain=sparse(newXtrain);
     Ytrain=sparse(newYtrain');
     Xtest=sparse(newXtest);
     Ytest=sparse(newYtest');
    
end

%% create a balanced set
function [Xbatch,Ybatch,idx]=createBalancedBatch(Xtrain,Ytrain,n_batch)
    
    [d,~]=size(Xtrain);
    [K,~]=size(Ytrain);
    Xbatch=zeros(d,n_batch);
    Ybatch=zeros(K,n_batch);
    
    u=cumsum(sum(Ytrain,2));
    sample=randsample((1:u(1)),n_batch/K);
    idx=[sample];
    Xbatch(:,1:n_batch/K)=Xtrain(:,sample);
    Ybatch(:,1:n_batch/K)=Ytrain(:,sample);
    for i=2:K
        sample=randsample((u(i-1):u(i)),n_batch/K);
        Xbatch(:,1+(i-1)*n_batch/K:i*n_batch/K)=Xtrain(:,sample);
        Ybatch(:,1+(i-1)*n_batch/K:i*n_batch/K)=Ytrain(:,sample);
        idx=[idx sample];
    end
    
    Xbatch=sparse(Xbatch);
    Ybatch=sparse(Ybatch);
    
end

%% Convolution matrices

function MF = MakeMFMatrix(F,nlen)

    [dd,k,nf]=size(F);
    
    vecF=reshape(F,[nf*dd*k,1]);
    vf=reshape(vecF,[dd*k,nf]);
    vf=vf';
        
    MF=zeros((nlen-k+1)*nf,nlen*dd);
        
    for i=1:nlen-k+1
        MF(1+nf*(i-1):nf*i,1+dd*(i-1):dd*(i-1)+dd*k)=vf;
    end
    
    MF=sparse(MF);

end

function MFs = buildMFs(convNet,nlen)

    F1=convNet.F{1};
    F2=convNet.F{2};
    [d,k1,nf1]=size(F1);
    [nf1,k2,nf2]=size(F2);
    
    MF1 = MakeMFMatrix(F1,nlen);
    MF2 = MakeMFMatrix(F2,nlen-k1+1);
    
    MFs={MF1;MF2};
    
end

function MXvec = MakeMXMatrix(x_input,d,k,nf)

    nlen=length(x_input)/d;
    X_input=reshape(x_input,[d,nlen]);
        
    MX=[];
        
    for i=1:nlen-k+1
        x=X_input(:,i:i+k-1);
        MX=[MX;kron(eye(nf),x(:)')];
    end
    
    MXvec=reshape(MX,[(nlen-k+1)*nf*k*nf*d,1]);
    MXvec=sparse(MXvec);

end

function MX1 = buildMX1(Xtrain,d,k1,nf1)
    
    [~,N]=size(Xtrain);
    MX1=sparse((19-k1+1)*nf1*k1*nf1*d,N);
    for i=1:N
        if (mod(i,1000)==0)
           disp(i/N); 
        end
        MX1(:,i)=MakeMXMatrix(Xtrain(:,i),d,k1,nf1);
    end
    MX1=sparse(MX1);
    
end

%% Evaluate Classifier

function [p,X1,X2] = EvaluateClassifier(X,MFs,W)

    MF1=MFs{1};
    MF2=MFs{2};
    X1=max(MF1*X,0);
    X2=max(MF2*X1,0);
    S_batch=W*X2;
    p=exp(S_batch)./sum(exp(S_batch),1);
    
end

%% Cost function

function loss = ComputeLoss(X_batch, Ys_batch, MFs, W)

    [~,size_batch]=size(X_batch);
    loss=0;
    for i=1:size_batch
        xi=X_batch(:,i);
        p=EvaluateClassifier(xi,MFs,W);
        loss=loss-log(Ys_batch(:,i)'*p);
    end
    loss=loss/size_batch;

end

function acc = ComputeAccuracy(X_batch, Ys_batch, MFs, W)

    [~,size_batch]=size(X_batch);
    acc=0;
    for i=1:size_batch
        xi=X_batch(:,i);
        p=EvaluateClassifier(xi,MFs,W);
        [~,predicted]=max(p);
        if Ys_batch(predicted,i)==1
            acc=acc+1;
        end
    end
    acc=acc/size_batch;

end
%% Gradient

function [gradW,gradF1,gradF2] = ComputeGradients(X,Y,convNet,MX1)

    [~,n]=size(X);
    F1=convNet.F{1};
    F2=convNet.F{2};
    W=convNet.W;
    [d,k1,nf1]=size(F1);
    [~,k2,nf2]=size(F2);
    MF = buildMFs(convNet,19);

    [P,X1,X2] = EvaluateClassifier(X,MF,W);
    
    G=-(Y-P);
    gradW=G*X2'/n;
    
    G=W'*G;
    G=G.*(X2>0);
    
    M=reshape(MakeMXMatrix(X1(:,1),nf1,k2,nf2),[((19-k1+1)-k2+1)*nf2,k2*nf2*nf1]);
    gradF2=zeros(size(G(:,1)'*M));
    for j=1:n
        gj=G(:,j);
        xj=X1(:,j);
        M=reshape(MakeMXMatrix(xj,nf1,k2,nf2),[((19-k1+1)-k2+1)*nf2,k2*nf2*nf1]);
        v=gj'*M;
        gradF2=gradF2+v/n;
    end

    G=MF{2}'*G;
    G=G.*(X1>0);
    
    M=reshape(MX1(:,1),[(19-k1+1)*nf1,k1*nf1*d]);
    gradF1=zeros(size(G(:,1)'*M));
    for j=1:n
        gj=G(:,j);
        M=reshape(MX1(:,j),[(19-k1+1)*nf1,k1*nf1*d]);
        v=gj'*M;
        gradF1=gradF1+v/n;
    end

end

function Gs = NumericalGradient(X_inputs, Ys, ConvNet, h)

try_ConvNet = ConvNet;
Gs = cell(length(ConvNet.F)+1, 1);

for l=1:length(ConvNet.F)
    try_convNet.F{l} = ConvNet.F{l};
    
    Gs{l} = zeros(size(ConvNet.F{l}));
    nf = size(ConvNet.F{l},  3);
    
    for i = 1:nf        
        try_ConvNet.F{l} = ConvNet.F{l};
        F_try = squeeze(ConvNet.F{l}(:, :, i));
        G = zeros(numel(F_try), 1);
        
        for j=1:numel(F_try)
            F_try1 = F_try;
            F_try1(j) = F_try(j) - h;
            try_ConvNet.F{l}(:, :, i) = F_try1; 

            MFs = buildMFs(try_ConvNet,19);
            l1 = ComputeLoss(X_inputs, Ys, MFs,try_ConvNet.W);

            F_try2 = F_try;
            F_try2(j) = F_try(j) + h;            
            
            try_ConvNet.F{l}(:, :, i) = F_try2;
            
            MFs = buildMFs(try_ConvNet,19);
            l2 = ComputeLoss(X_inputs, Ys, MFs,try_ConvNet.W);
            
            G(j) = (l2 - l1) / (2*h);
            try_ConvNet.F{l}(:, :, i) = F_try;
        end
        
        Gs{l}(:, :, i) = reshape(G, size(F_try));
    end
end

% compute the gradient for the fully connected layer
W_try = ConvNet.W;
G = zeros(numel(W_try), 1);
for j=1:numel(W_try)
    W_try1 = W_try;
    W_try1(j) = W_try(j) - h;
    try_ConvNet.W = W_try1; 
            
    MFs = buildMFs(try_ConvNet,19);
    l1 = ComputeLoss(X_inputs, Ys, MFs,try_ConvNet.W);
            
    W_try2 = W_try;
    W_try2(j) = W_try(j) + h;            
            
    try_ConvNet.W = W_try2;
    
    MFs = buildMFs(try_convNet,19);
    l2 = ComputeLoss(X_inputs, Ys, MFs,try_ConvNet.W);
            
    G(j) = (l2 - l1) / (2*h);
    try_ConvNet.W = W_try;
end
Gs{end} = reshape(G, size(W_try));

end

       
%%
function [Convstar,res] = MiniBatchGD(Xtrain,Ytrain,Xtest,Ytest,GDparams,ConvNet,MX1)

    n_batch=GDparams(1);
    eta=GDparams(2);
    n_epochs=GDparams(3);
    rho=GDparams(4);
    
    nlen=19;
    
    update=0;
    
    %hold on
    
    [~,N]=size(Xtrain);
    vF1=zeros(size(ConvNet.F{1}));
    vF2=zeros(size(ConvNet.F{2}));
    vW=zeros(size(ConvNet.W));
    res=zeros(n_epochs/10,1);

    for t=1:n_epochs
        MF=buildMFs(ConvNet,nlen);

        % take a sample
        [Xbatch,Ybatch,idx]=createBalancedBatch(Xtrain,Ytrain,n_batch);
      %  idx=randsample((1:N),n_batch);
      %  Xbatch=Xtrain(:,idx);
      %  Ybatch=Ytrain(:,idx);
        
        % update the network
        MX1batch=MX1(:,idx);
        [gradW,gradF1,gradF2] = ComputeGradients(Xbatch,Ybatch,ConvNet,MX1batch);
        gradF1=reshape(gradF1,size(ConvNet.F{1}));
        gradF2=reshape(gradF2,size(ConvNet.F{2}));
        vF1=rho*vF1+eta*gradF1;
        vF2=rho*vF2+eta*gradF2;
        vW=rho*vW+eta*gradW;
        ConvNet.F{1}=ConvNet.F{1}-vF1;
        ConvNet.F{2}=ConvNet.F{2}-vF2;
        ConvNet.W=ConvNet.W-vW;
        update=update+1;
        
        % Results
        if mod(update,25)==0
            %disp(ComputeLoss(Xtest,Ytest,MF,ConvNet.W))
            res(update/25,1)=ComputeLoss(Xtest,Ytest,MF,ConvNet.W);
        end
        if mod(update,100)==0
            disp(update/n_epochs)
            disp(100*ComputeAccuracy(Xtest,Ytest,MF,ConvNet.W))
        end

    end
    
    Convstar.F{1}=ConvNet.F{1};
    Convstar.F{2}=ConvNet.F{2};
    Convstar.W=ConvNet.W;
end
    
