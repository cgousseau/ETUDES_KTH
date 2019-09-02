function rotation(F,alpha)

    subplot(1,3,1)
    G = rot(F,alpha);
    showgrey(G)
    title(sprintf('G : image after rotation by alpha'))
    Ghat = fft2(G);
    subplot(1,3,2)
    showfs(Ghat)
    title(sprintf('Ghat : fft of G'))
    Hhat = rot(fftshift(Ghat), -alpha );
    subplot(1,3,3)
    showgrey(log(1 + abs(Hhat)))
    title(sprintf('Hhat : fft of Ghat after rotation by -alpha'))
    
end

