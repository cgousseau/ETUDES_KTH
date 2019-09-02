function linearity(F,G,H)
    
    subplot(3,3,1);
    showgrey(F);
    title(sprintf('F'))
    subplot(3,3,2);
    showgrey(G);
    title(sprintf('G = transpose(F)'))
    subplot(3,3,3);
    showgrey(H);
    title(sprintf('H = F+2*G'))

    Fhat = fft2(F);
    Ghat = fft2(G);
    Hhat = fft2(H);

    subplot(3,3,4);
    showgrey(log(1 + abs(Fhat)));
    title(sprintf('log(1 + abs(Fhat))'))
    subplot(3,3,5);
    showgrey(log(1 + abs(Ghat)));
    title(sprintf('log(1 + abs(Ghat))'))
    subplot(3,3,6);
    showgrey(log(1 + abs(Hhat)));
    title(sprintf('log(1 + abs(Hhat))'))

    subplot(3,3,7);
    showgrey(log(1 + abs(fftshift(Hhat))));
    title(sprintf('log(1 + abs(fftshift(Hhat)))'))
    
end

