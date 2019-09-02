function multiplication(F,G)

    subplot(1,3,1)
    showgrey(F.*G);
    title(sprintf('F.*G'))
    subplot(1,3,2)
    showfs(fft2(F.*G));
    title(sprintf('fft(F*G)'))
    subplot(1,3,3)
    Fhat = fft2(F);
    Ghat = fft2(G);
    showfs(fftshift(conv2(fftshift(Fhat), fftshift(Ghat), 'same')/(128*128)));
    title(sprintf('conv2(fft(F),fft(G))'))
    
end

