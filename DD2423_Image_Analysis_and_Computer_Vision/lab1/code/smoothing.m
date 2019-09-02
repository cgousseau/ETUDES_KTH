function smoothing(img)

    add=img;

    % with gaussfft
    subplot(3,4,1)
    showgrey(add)
    title(sprintf('original image'))
    subplot(3,4,2)
    showgrey(gaussfft(add,1))
    title(sprintf('gaussfft, t=1'))
    subplot(3,4,3)
    showgrey(gaussfft(add,4))
    title(sprintf('gaussfft, t=4'))
    subplot(3,4,4)
    showgrey(gaussfft(add,16))
    title(sprintf('gaussfft, t=16'))
    
    % with medfilt
    subplot(3,4,5)
    showgrey(add)
    title(sprintf('original image'))
    subplot(3,4,6)
    showgrey(medfilt(add,3))
    title(sprintf('medfilt, 3x3'))
    subplot(3,4,7)
    showgrey(medfilt(add,5))
    title(sprintf('medfilt, 5x5'))
    subplot(3,4,8)
    showgrey(medfilt(add,7))
    title(sprintf('medfilt, 7x7'))
    
    % with ideal
    subplot(3,4,9)
    showgrey(add)
    title(sprintf('original image'))
    subplot(3,4,10)
    showgrey(ideal(add,0.3))
    title(sprintf('ideal, CUTOFF = 0.3'))
    subplot(3,4,11)
    showgrey(ideal(add,0.1))
    title(sprintf('ideal,  CUTOFF = 0.1'))
    subplot(3,4,12)
    showgrey(ideal(add,0.05))
    title(sprintf('ideal,  CUTOFF = 0.05'))
    
end

