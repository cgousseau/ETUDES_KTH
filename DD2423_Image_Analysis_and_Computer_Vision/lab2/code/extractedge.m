function edgecurves = extractedge(inpic, scale, threshold, shape)%UNTITLED5 Summary of this function goes here
    im=discgaussfft(inpic,scale);
    lvv=Lvvtilde(im,shape);
    lvvv=Lvvvtilde(im,shape);
    k1=zerocrosscurves(lvv,lvvv<0);
    dxmask=[-1,0,1];
    dymask=[-1;0;1];
    Lx = filter2(dxmask,inpic,shape);
    Ly = filter2(dymask,inpic,shape);
    gradmagn = Lx.^2 + Ly.^2;
    gradmagn=gradmagn/max(max(gradmagn))*255;
    edgecurves=thresholdcurves(k1,(gradmagn-threshold>0)-1);
end

