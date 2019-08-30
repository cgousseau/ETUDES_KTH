
%% Read the data

book_fname = 'goblet_book.txt';
fid = fopen(book_fname,'r');
book_data = fscanf(fid,'%c');
fclose(fid);

book_chars=unique(book_data);
K=length(book_chars);

% Create the maps
keys=split(book_chars,'');
keys=keys(2:end-1);
val=(1:length(keys));
char_to_ind=containers.Map(keys,val);
ind_to_char=containers.Map(val,keys);

%% Set hyper-parameters & initialize the RNN’s parameters

m=100;
eta=0.1;
seq_length=25;
RNN.b=randn(m,1);
RNN.c=randn(K,1);
sig=0.01;
RNN.U = randn(m,K)*sig;
RNN.W = randn(m,m)*sig;
RNN.V = randn(K,m)*sig;

%% create a random text
h0=zeros(m,1);
x0=randn(K,1);
n=10;
Y = createText(RNN,h0,x0,n,K);
txt=matrix_to_text(Y,ind_to_char);
disp(txt);

%% check the gradients 
X_chars = book_data(1:seq_length);
Y_chars = book_data(2:seq_length+1);

X=text_to_matrix(X_chars,char_to_ind,K);
Y=text_to_matrix(Y_chars,char_to_ind,K);

h0=zeros(m,1);
num_grads = ComputeGradsNum(X, Y, RNN, 0.0001);
grads = ComputeGrads(X, Y, RNN,h0);

disp('max relative error wrt to U:');
disp(max(max(abs(num_grads.U-grads.U)./grads.U)));
disp('max relative error wrt to V:');
disp(max(max(abs(num_grads.V-grads.V)./grads.V)));
disp('max relative error wrt to W:');
disp(max(max(abs(num_grads.W-grads.W)./grads.W)));
disp('max relative error wrt to b:');
disp(max(abs(num_grads.b-grads.b)./grads.b));
disp('max relative error wrt to c:');
disp(max(abs(num_grads.c-grads.c)./grads.c));


%% Training
%X=text_to_matrix(book_data,char_to_ind,K);
epochs=300000;
eta=0.1;
seq_length=25;
[finalRNN,res] = run(RNN,epochs,X,seq_length,m,eta,ind_to_char);

%%
% create a random text
h0=zeros(m,1);
x0=randn(K,1);
n=1000;
Y = createText(finalRNN,h0,x0,n,K);
txt=matrix_to_text(Y,ind_to_char);
%% Synthesize text from your randomly initialized RNN

function Y = createText(RNN,h0,x0,n,K)
    
    Y=zeros(K,n);
    h_old=h0;
    x=x0;
    for i=1:n
        ai=RNN.W*h_old+RNN.U*x+RNN.b;
        h_new=tanh(ai);
        oi=RNN.V*h_new+RNN.c;
        pi=exp(oi)/sum(exp(oi));
        cpi=cumsum(pi);
        a=rand;
        ixs=find(cpi-a>0);
        ii=ixs(1);
        Y(ii,i)=1;
        h_old=h_new;
        x=zeros(K,1);
        x(ii)=1;
    end
        
end

%% matrix from/to text

function txt=matrix_to_text(Y,ind_to_char)

    [~,n]=size(Y);
    txt=char(n);
    for i=1:n
        [~,idx]=max(Y(:,i));
        txt(i)=ind_to_char(idx);
    end
    
end

function mat=text_to_matrix(txt,char_to_ind,K)

    [~,n]=size(txt);
    mat=zeros(K,n);
    for i=1:n
        disp(i/n);
        mat(char_to_ind(txt(i)),i)=1;
    end
    
end

%% forward pass

function [a,h,o,p,loss] = ComputeLoss(x,y,RNN,h0)
    
%     y=zeros(RNN.K,1);
    
    h=h0;
    [~,n]=size(x);
    loss=0;
    
    for t=1:n  
        xt=x(:,t);
        yt=y(:,t);
        a(:,t)=RNN.W*h(:,t)+RNN.U*xt+RNN.b;
        h(:,t+1)=tanh(a(:,t));
        o(:,t)=RNN.V*h(:,t+1)+RNN.c;
        p(:,t)=exp(o(:,t))/sum(exp(o(:,t)));
        
        loss=loss-log(yt'*p(:,t));
                
    end
           
end

%% Gradients computation

function num_grads = ComputeGradsNum(X, Y, RNN, h)

for f = fieldnames(RNN)'
    disp('Computing numerical gradient for')
    disp(['Field name: ' f{1} ]);
    num_grads.(f{1}) = ComputeGradNumSlow(X, Y, f{1}, RNN, h);
end

end

function grad = ComputeGradNumSlow(X, Y, f, RNN, h)

n = numel(RNN.(f));
grad = zeros(size(RNN.(f)));
hprev = zeros(size(RNN.W, 1), 1);
for i=1:n
    RNN_try = RNN;
    RNN_try.(f)(i) = RNN.(f)(i) - h;
    [~,~,~,~,l1] = ComputeLoss(X, Y, RNN_try, hprev);
    RNN_try.(f)(i) = RNN.(f)(i) + h;
    [~,~,~,~,l2] = ComputeLoss(X, Y, RNN_try, hprev);
    grad(i) = (l2-l1)/(2*h);
end

end

function grads = ComputeGrads(X, Y, RNN,h0)

for f = fieldnames(RNN)'
%     disp('Computing analytical gradient for')
%     disp(['Field name: ' f{1} ]);
    grads.(f{1}) = ComputeGrad(X, Y, f{1}, RNN,h0);
end

end

function grad = ComputeGrad(X, Y, f, RNN,h0)

    grad = zeros(size(RNN.(f)));
    [~,n]=size(X);
    
    [a,h,o,p,loss] = ComputeLoss(X,Y,RNN,h0);
    
    if f=='V'
        for t=1:n
            yt=Y(:,t);
            pt=p(:,t);
            gt=-(yt-pt)';
            ht=h(:,t+1);
            grad=grad+gt'*ht';
        end
    elseif f=='W'
        for t=1:n
            yt=Y(:,t);
            pt=p(:,t);
            dL_do(t,:)=-(yt-pt)';
        end 
        dL_dh(n,:)=dL_do(n,:)*RNN.V;
        dL_da(n,:)=dL_dh(n,:)*diag(1-power(tanh(a(:,n)),2));
        for t=flip(1:n-1)
            dL_dh(t,:)=dL_do(t,:)*RNN.V+dL_da(t+1,:)*RNN.W;
            dL_da(t,:)=dL_dh(t,:)*diag(1-power(tanh(a(:,t)),2));
        end
        for t=1:n
            gt=dL_da(t,:);
            ht=h(:,t);
            grad=grad+gt'*ht';
        end
    elseif f=='U'
        for t=1:n
            yt=Y(:,t);
            pt=p(:,t);
            dL_do(t,:)=-(yt-pt)';
        end 
        dL_dh(n,:)=dL_do(n,:)*RNN.V;
        dL_da(n,:)=dL_dh(n,:)*diag(1-power(tanh(a(:,n)),2));
        for t=flip(1:n-1)
            dL_dh(t,:)=dL_do(t,:)*RNN.V+dL_da(t+1,:)*RNN.W;
            dL_da(t,:)=dL_dh(t,:)*diag(1-power(tanh(a(:,t)),2));
        end
        for t=1:n
            gt=dL_da(t,:);
            xt=X(:,t);
            grad=grad+gt'*xt';
        end
    elseif f=='b'
        for t=1:n
            yt=Y(:,t);
            pt=p(:,t);
            dL_do(t,:)=-(yt-pt)';
        end 
        dL_dh(n,:)=dL_do(n,:)*RNN.V;
        dL_da(n,:)=dL_dh(n,:)*diag(1-power(tanh(a(:,n)),2));
        for t=flip(1:n-1)
            dL_dh(t,:)=dL_do(t,:)*RNN.V+dL_da(t+1,:)*RNN.W;
            dL_da(t,:)=dL_dh(t,:)*diag(1-power(tanh(a(:,t)),2));
        end
        for t=1:n
            gt=dL_da(t,:);
            grad=grad+gt';
        end
    elseif f=='c'
        for t=1:n
            yt=Y(:,t);
            pt=p(:,t);
            dL_do(t,:)=-(yt-pt)';
        end 
        for t=1:n
            gt=dL_do(t,:);
            grad=grad+gt';
        end
    end
end

%% Training

function [finalRNN,res] = run(RNN,epochs,X,seq_length,m,eta,ind_to_char)
    
    e=1;
    h0=zeros(m,1);
    mU=zeros(size(RNN.U));
    mV=zeros(size(RNN.V));
    mW=zeros(size(RNN.W));
    mb=zeros(size(RNN.b));
    mc=zeros(size(RNN.c));
    res=[];
    
    fid=fopen('results.txt','wt');
    
    for iter=1:epochs
        
        Xtrain=X(:,e:e+seq_length-1);
        Ytrain=X(:,e+1:e+seq_length);
        grads = ComputeGrads(Xtrain, Ytrain, RNN,h0);
        
        mU=mU+grads.U.*grads.U;
        mV=mV+grads.V.*grads.V;
        mW=mW+grads.W.*grads.W;
        mb=mb+grads.b.*grads.b;
        mc=mc+grads.c.*grads.c;
        
        RNN.U=RNN.U-eta./sqrt(mU+0.0000001).*grads.U;
        RNN.V=RNN.V-eta./sqrt(mV+0.0000001).*grads.V;
        RNN.W=RNN.W-eta./sqrt(mW+0.0000001).*grads.W; 
        RNN.b=RNN.b-eta./sqrt(mb+0.0000001).*grads.b; 
        RNN.c=RNN.c-eta./sqrt(mc+0.0000001).*grads.c; 
        
        [~,h,~,~,loss] = ComputeLoss(Xtrain,Ytrain,RNN,h0);
        
        if e==1
            smooth_loss=loss;
        else
            smooth_loss=0.999*smooth_loss+0.001*loss;
        end
        
        h0=h(:,seq_length+1);
        e=e+seq_length;
        
        if (mod(iter,500)==0)
            disp(iter);
            disp(smooth_loss)
            res=[res smooth_loss];
        end 
        
        if (mod(iter,10000)==0)
            y=createText(RNN,h0,Xtrain(:,seq_length),200,83);
            txt=matrix_to_text(y,ind_to_char);
            fprintf(fid, '%s %f %s %f \n','iter=',iter,'loss=',smooth_loss);
            fprintf(fid, '%s\n',txt);
        end
        
        if e>1107000
            e=1;
            h0=zeros(m,1);
        end
        
    end
    
    fclose(fid);

    
    finalRNN=RNN;

end