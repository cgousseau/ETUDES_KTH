function [linepar,acc] = houghedgeline(pic, scale,gradmagnthreshold,nrho,ntheta,nlines,verbose)

    edges = extractedge(pic,scale,gradmagnthreshold,'same');
    
    sz=size(pic,2);
    edges=min(edges,sz);
    %rho=linspace(0,sqrt(2)*sz,nrho);
    drho=sqrt(2)*sz/nrho;
    %theta=linspace(0,pi/2,ntheta);
    dtheta=pi/ntheta;
    acc=zeros(nrho,ntheta);
    
    dxmask=[-1,0,1];
    dymask=[-1;0;1];
    Lx = filter2(dxmask,pic,'same');
    Ly = filter2(dymask,pic,'same');
    gradmagn = Lx.^2 + Ly.^2;
    gradmagn=gradmagn/max(max(gradmagn));
    
    for i=1:length(edges)
        %disp(i/length(edges))
        xi=edges(1,i); 
        yi=sz-edges(2,i);
        if xi>0 && xi<sz && yi>0 &&yi<sz
            gi=gradmagn(round(edges(2,i)),round(edges(1,i)));
            %acc=acc+log(1+gi)*linexy(xi,yi,ntheta,nrho,sz);
            %acc=acc+exp(1+gi)*linexy(xi,yi,ntheta,nrho,sz);
            %acc=acc+gi*linexy(xi,yi,ntheta,nrho,sz);
            acc=acc+linexy(xi,yi,ntheta,nrho,sz);
        end
    end  
    
    [pos,value] = locmax8(acc);
    [~,indexvector] = sort(value);
    nmaxima = size(value, 1);
    
    linepar=zeros(2,nlines);
    for idx = 1:nlines
        linepar(1,idx) = pos(indexvector(nmaxima - idx + 1), 1)*drho-drho;
        linepar(2,idx) = pos(indexvector(nmaxima - idx + 1), 2)*dtheta-pi/2;
    end

end
