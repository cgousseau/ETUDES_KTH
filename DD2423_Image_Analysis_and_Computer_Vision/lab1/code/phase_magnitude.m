function phase_magnitude(img1,img2,img3)

    subplot(3,3,1)
    showgrey(img1);
    subplot(3,3,2)
    showgrey(pow2image(img1,10^-10));
    title(sprintf('with the same phase'))
    subplot(3,3,3)
    showgrey(randphaseimage(img1));
    title(sprintf('with the same magnitude'))
    subplot(3,3,4)    
    showgrey(img2);
    subplot(3,3,5)
    showgrey(pow2image(img2,10^-10));
    subplot(3,3,6)
    showgrey(randphaseimage(img2));
    subplot(3,3,7)    
    showgrey(img3);
    subplot(3,3,8)    
    showgrey(pow2image(img3,10^-10));
    subplot(3,3,9)
    showgrey(randphaseimage(img3));
    
end

