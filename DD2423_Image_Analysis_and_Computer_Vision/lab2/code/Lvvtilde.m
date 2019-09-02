function res = Lvvtilde(inpic,shape)
    if (nargin<2)
        shape='same';
    end
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
    dxy=conv2(dx,dy,'same');
    Lx=filter2(dx,inpic,shape);
    
    Ly=filter2(dy,inpic,shape);
    Lxx=filter2(dxx,inpic,shape);
    Lyy=filter2(dyy,inpic,shape);
    Lxy=filter2(dxy,inpic,shape);
    res=Lx.*Lx.*Lxx+2*Lx.*Ly.*Lxy+Ly.*Ly.*Lyy;
end