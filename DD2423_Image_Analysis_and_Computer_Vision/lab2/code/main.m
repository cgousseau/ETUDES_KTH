% LAB2

%% Question 1
tools=few256;
deltax=[-1,0,1];
deltay=[-1;0;1];
dxtools = conv2(tools, deltax, 'valid');
dytools = conv2(tools, deltay, 'valid');
subplot(1,3,1);
showgrey(tools);
title('original image')
subplot(1,3,2)
showgrey(dxtools)
title('x derivative')
subplot(1,3,3)
showgrey(dytools)
title('y derivative')

%% Question 2&3 1
dxtools_reshaped=dxtools(2:255,:);
dytools_reshaped=dytools(:,2:255);
gradmagntools=sqrt(dxtools_reshaped.^2+dytools_reshaped.^2);
subplot(1,2,1)
showgrey(gradmagntools);
title('magnitude of the gradient')
subplot(1,2,2)
threshold=35;
showgrey((gradmagntools - threshold) > 0)
title('magnitude of the gradient after thresholding')

%% Question 2&3 2
subplot(2,2,1)
showgrey(godthem256)
title('original image')
dxmask=[-1,0,1];
dymask=[-1;0;1];
Lx = filter2(dxmask, godthem256, 'same');
Ly = filter2(dymask, godthem256, 'same');
pixels = Lx.^2 + Ly.^2;
subplot(2,2,2)
threshold=1800;
showgrey((pixels - threshold) > 0)
title('thresholded gradient magnitude without smoothing')
subplot(2,2,3)
godthem256_smoothed=gaussfft(godthem256,1);
showgrey(godthem256_smoothed)
title('original image after smoothing')
subplot(2,2,4)
Lx = filter2(dxmask, godthem256_smoothed, 'same');
Ly = filter2(dymask, godthem256_smoothed, 'same');
pixels = Lx.^2 + Ly.^2;
threshold=800;
showgrey((pixels - threshold) > 0)
title('thresholded gradient magnitude with smoothing')

%% Question 4
dx=[0,0,0,0,0;
    0,0,0,0,0;
    0,-0.5,0,0.5,0;
    0,0,0,0,0;
    0,0,0,0,0];
dy=[0,0,0,0,0;
    0,0,-0.5,0,0;
    0,0,0,0,0;
    0,0,0.5,0,0;
    0,0,0,0,0];
dxx=[0,0,0,0,0;
    0,0,0,0,0;
    0,1,-2,1,0;
    0,0,0,0,0;
    0,0,0,0,0];
dyy=[0,0,0,0,0;
    0,0,1,0,0;
    0,0,-2,0,0;
    0,0,1,0,0;
    0,0,0,0,0];
dxxx=conv2(dx,dxx,'same');
dxy=conv2(dx,dy,'same');
dxxy=conv2(dxx,dy,'same');

vect=[0.0001;1;4;16;64];
subplot(1,6,1)
house=godthem256;
showgrey(house)
title('original image')
for i=1:5
    t=vect(i);
    subplot(1,6,1+i)
    C = contour(Lvvtilde(discgaussfft(house,t),'same'),[0 0]);
    axis('image')
    axis('ij')
    title(sprintf('Lvvtilde, t=%f',t))
end
%%
t=4;
subplot(1,3,1)
house=godthem256;
showgrey(house)
title('original image')
subplot(1,3,2)
C = contour(Lvvtilde(discgaussfft(house,t),'same'),[0 0]);
axis('image')
axis('ij')
title(sprintf('Lvvtilde, t=%f',t))
subplot(1,3,3)
lvvv=(Lvvvtilde(discgaussfft(house,t),'same')<0);
showgrey(1-lvvv)
title(sprintf('Lvvvtilde, t=%f',t))

%% Question 5
x=C(1,:);
y=C(2,:);
y=min(y,255);
Cim=ones(256,256);
for i=1:length(x)
    Cim(round(1+y(i)),round(1+x(i)))=0;
end
subplot(1,2,1)
showgrey(Cim)
combined=ones(256,256);
for i=1:256
    for j=1:256
        if Cim(i,j)==0
            if lvvv(i,j)==0
                combined(i,j)=0;
            end
        end
    end
end
subplot(1,2,2)
showgrey(combined)

%% Question 6
t=10;
lvv=Lvvtilde(discgaussfft(house,t),'same');
lvvv=Lvvvtilde(discgaussfft(house,t),'same');
k=zerocrosscurves(lvv,-lvvv);
subplot(1,3,1)
showgrey(house)
title('original image')
subplot(1,3,2)
m = contour(lvv,[0 0]);
overlaycurves(house,m);
title('lvv=0')
subplot(1,3,3)
overlaycurves(house,k)
title('lvv=0 & lvvv<0')

%% Question 7.1
scale=10;
threshold=3;
edgecurves = extractedge(tools,scale,threshold,'same');
overlaycurves(tools,edgecurves)
title(sprintf('scale=%i, threshold=%i',scale,threshold))
%% Question 7.2
scale=10;
threshold=2;
edgecurves = extractedge(godthem256,scale,threshold,'same');
overlaycurves(godthem256,edgecurves)
title(sprintf('scale=%i, threshold=%i',scale,threshold))

%% Question 8-9-10
im=triangle128;
scale=0.00001;
threshold=0;
shape='same';
[linepar,acc] = houghedgeline(im,scale,threshold,1000,1000,5);
imshow(mat2gray(acc))
%%
sz=size(im,2);
newim=zeros(sz,sz);
for i=1:sz
    for j=1:sz
        newim(i,j)=im(1+sz-i,j);
    end
end
hold on
showgrey(newim)
axis('xy')
for i=1:nlines
    rho_i=linepar(1,i);
    theta_i=linepar(2,i);
    t=linspace(-1000,1000,10);
    x=rho_i*cos(theta_i)-t*sin(theta_i);
    y=rho_i*sin(theta_i)+t*cos(theta_i);
    plot(x,y);
    axis([0 sz 0 sz])
end

