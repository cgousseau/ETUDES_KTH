function psf = gaussfft(pic, t)

    % Generate a filter based on a sampled version of the Gaussian function
    sz=size(pic,1);
    [X,Y]=meshgrid(-sz/2:sz/2-1,-sz/2:sz/2-1);
    gaussianFilter=1/(2*pi*t)*exp(-(X.*X+Y.*Y)/(2*t));
    
    % Fourier transform the original image and the Gaussian filter
    picHat=fft2(pic);
    %picHat=fftshift(picHat);
    gaussianFilterHat=fftshift(gaussianFilter);
    gaussianFilterHat=fft2(gaussianFilterHat);
    
    % Multiply the Fourrier transforms
    prod=picHat.*gaussianFilterHat;
    
    % Invert the Fourrier transform
    results=ifft2(prod);
    
    % return
    psf = real(results);