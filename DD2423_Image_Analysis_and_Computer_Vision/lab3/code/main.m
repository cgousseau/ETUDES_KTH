
%% K-means clustering
image=imread('tiger1.jpg');
K=[5;15];
L=500;
height=size(image,1);
width=size(image,2);
subplot(1,3,1)
imshow(image)
title(sprintf('original image'))
for i=1:length(K)
    k=K(i);
    Ivec = reshape(double(image), width*height, 3);
    [s,c] = kmeans_segm(Ivec,k,L,0);

    newimvec=zeros(size(image,1)*size(image,2),3);
    for j=1:size(image,1)*size(image,2)
        newimvec(j,:)=c(s(j),:);
    end
    newim = reshape(uint8(newimvec), size(image,1),size(image,2), 3);
    subplot(1,3,i+1)
    imshow(newim)
    title(sprintf('k-means, k=%d',k))
end

%% Mean-shift segmentation

scale_factor = 0.5;       % image downscale factor
spatial_bandwidth = 100.0;  % spatial bandwidth
colour_bandwidth = 1.0;   % colour bandwidth
num_iterations = 40;      % number of mean-shift iterations
image_sigma = 1.0;        % image preblurring scale

I = imread('tiger1.jpg');
I = imresize(I, scale_factor);
Iback = I;
d = 2*ceil(image_sigma*2) + 1;
h = fspecial('gaussian', [d d], image_sigma);
I = imfilter(I, h);

segm = mean_shift_segm(I, spatial_bandwidth, colour_bandwidth, num_iterations);
Inew = mean_segments(Iback, segm);
I = overlay_bounds(Iback, segm);
subplot(1,2,1); imshow(Inew);
subplot(1,2,2); imshow(I);
 
%% Normalized cut
colour_bandwidth = 20.0; % color bandwidth
radius = 30;              % maximum neighbourhood distance

ncuts_thresh = 0.8;      % cutting threshold
min_area = 20;          % minimum area of segment
max_depth = 8;           % maximum splitting depth

scale_factor = 0.4;      % image downscale factor
image_sigma = 2.0;       % image preblurring scale

I = imread('tiger1.jpg');
I = imresize(I, scale_factor);
Iback = I;
d = 2*ceil(image_sigma*2) + 1;
h = fspecial('gaussian', [d d], image_sigma);
I = imfilter(I, h);

segm = norm_cuts_segm(I, colour_bandwidth, radius, ncuts_thresh, min_area, max_depth);
Inew = mean_segments(Iback, segm);
I = overlay_bounds(Iback, segm);
subplot(1,2,1); imshow(Inew);
subplot(1,2,2); imshow(I);

%% Segmentation using graph cuts
scale_factor = 0.5;          % image downscale factor
area = [ 80, 110, 570, 300 ]; % image region to train foreground with
K = 1;                      % number of mixture components
alpha = 8.0;                 % maximum edge cost
sigma = 8.0;                % edge cost decay factor

I = imread('tiger1.jpg');
I = imresize(I, scale_factor);
Iback = I;
area = int16(area*scale_factor);
[ segm, prior ] = graphcut_segm(I, area, K, alpha, sigma);
Inew = mean_segments(Iback, segm);
I = overlay_bounds(Iback, segm);
subplot(2,2,1); imshow(Inew);
subplot(2,2,2); imshow(I);
subplot(2,2,3); imshow(prior);
