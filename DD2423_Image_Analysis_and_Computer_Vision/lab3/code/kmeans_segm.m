function [ segmentation, centers ] = kmeans_segm(Ivec, K, L, seed)


    % Randomly initialize the K cluster centers
    %centers=255*rand(K,3);
    centers=Ivec(randi(size(Ivec,1),K,1),:);
    
    idx_min_old=zeros(size(Ivec,1),1);
    for i=1:L
      dist=pdist2(Ivec,centers);
      [~,idx_min]=min(dist,[],2);
      for j=1:K
          centers(j,:)=mean(Ivec(idx_min==j,:));
      end
      centers(isnan(centers))=0;
      if isequal(idx_min_old,idx_min)
          break;
      end
      %disp(sum(idx_min_old==idx_min)/size(Ivec,1))
      idx_min_old=idx_min;
    end
    
    sprintf('=== CONVERGED AFTER %d ITERATIONS ===',i)
    dist=pdist2(Ivec,centers);    
    [~,segmentation]=min(dist,[],2);

end

