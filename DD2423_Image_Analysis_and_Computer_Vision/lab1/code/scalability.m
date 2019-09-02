function scalability(F,G)

    subplot(2,2,1)
    showgrey(F.*G);
    title(sprintf('F.*G'))
    subplot(2,2,2)
    showfs(fft2(F.*G));
    title(sprintf('fft(F*G)'))

    F = [zeros(60, 128); ones(8, 128); zeros(60, 128)] .* [zeros(128, 48) ones(128, 32) zeros(128, 48)];
    subplot(2,2,3)
    showgrey(F)
    Fhat=fft2(F);
    subplot(2,2,4)
    showfs(Fhat)
    
end

