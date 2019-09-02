function mat=linexy(x,y,ntheta,nrho,sz)
    mat=zeros(nrho,ntheta);
    dtheta=pi/ntheta;
    drho=sqrt(2)*sz/nrho;
    for j=1:ntheta
        theta=-pi/2+j*dtheta;
        rho=x*cos(theta)+y*sin(theta);
        if rho>=0 && round(rho/drho)<nrho-1
            mat(1+round(rho/drho),j)=1;
        end
    end
end