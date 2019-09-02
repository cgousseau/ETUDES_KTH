%% Basis functions
fftwave(63,1,128)

%% Linearity
F = [ zeros(56, 128); ones(16, 128); zeros(56, 128)];
G = F';
H = F+2*G;

linearity(F,G,H)

%% Multiplication

F = [ zeros(56, 128); ones(16, 128); zeros(56, 128)];
G = F';

multiplication(F,G)

%% Scalability

F = [zeros(60, 128); ones(8, 128); zeros(60, 128)] .* [zeros(128, 48) ones(128, 32) zeros(128, 48)];
G = F';

scalability(F,G)

%% Rotation

alpha=30;
F = [zeros(60, 128); ones(8, 128); zeros(60, 128)] .* [zeros(128, 48) ones(128, 32) zeros(128, 48)];
rotation(F,alpha)


%% Information in Fourier phase and magnitude

img1=phonecalc128;
img2=few128;
img3=nallo128;

phase_magnitude(img1,img2,img3)

%%  Gaussian convolution implemented via FFT

vect=[0.1, 0.3, 1, 10 , 100];
for i=1:5
    t=vect(i);
    psf=gaussfft(deltafcn(128, 128), t);
    subplot(1,5,i)
    showgrey(psf)
    title(sprintf('t = %f', t))
    sprintf('t = %f',t)
    var=variance(psf)
end

%%
img1=few256;
img2=godthem256;
subplot(2,6,1)
showgrey(img1)
title(sprintf('original image'))
subplot(2,6,7)
showgrey(img2)
title(sprintf('original image'))
vect=[1.0, 4.0, 16.0, 64.0, 256.0];
for i=1:5
    t=vect(i);
    psf1=gaussfft(img1, t);
    subplot(2,6,1+i)
    showgrey(psf1)
    title(sprintf('t = %i', t))
    psf2=gaussfft(img2, t);
    subplot(2,6,1+i+6)
    showgrey(psf2)
    title(sprintf('t = %i', t))
end

%% Smoothing of noisy data - gaussian

office = office256;
add = gaussnoise(office, 16);

smoothing(add)

%% Smoothing of noisy data - salt and pepper

sap = sapnoise(office, 0.1, 255);

smoothing(sap)

%% Smoothing and subsampling

vect=[4;3;2;1;1];
img = phonecalc256;
smoothimg = img;
N=4;
for i=1:N
    if i>1 % generate subsampled versions
        img = rawsubsample(img);
        smoothimg = gaussfft(smoothimg, t);
        smoothimg = rawsubsample(smoothimg);
    end
    subplot(2, N, i)
    showgrey(img)
    subplot(2, N, i+N)
    showgrey(smoothimg)
end

%%
img = phonecalc256;
smoothimg = img;
N=4;
for i=1:N
    if i>1 % generate subsampled versions
        img = rawsubsample(img);
        smoothimg = ideal(smoothimg, 0.3);
        smoothimg = rawsubsample(smoothimg);
    end
    subplot(2, N, i)
    showgrey(img)
    subplot(2, N, i+N)
    showgrey(smoothimg)
end

