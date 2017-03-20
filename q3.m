function q3(im_name)
close all;
clear;
clusters = 7;

img = imread(im_name);
img = imresize(img,0.1);
img = img(70:320,100:470,:);
img_in = img;

img = rgb2lab(img);
[im_height,im_width,~] = size(img);
load('gmm_db.mat');

gmm_fore_db = {gmm_orange, gmm_cleaned, gmm_s1, gmm_white, gmm_blue, ...
    gmm_2, ...
    gmm_3, ...
    gmm_4, ...
    gmm_dark_blue, gmm_green};
gmm_back_db = {gmm_orange_back, gmm_cleaned_back, gmm_s1_back, ...
    gmm_2_back, ...
    gmm_3_back, ...
    gmm_4_back, ...
    gmm_white_back, gmm_blue_back, gmm_dark_blue_back, gmm_green_back};

% features = ComputePositionColorFeatures(img);
% features = features(:,:,:);
im_vec = reshape(permute(img,[3 1 2]), [], im_height * im_width)';
im_in_vec = reshape(permute(img_in,[3 1 2]), [], im_height * im_width)';
img_temp = im_in_vec;
R = zeros(im_height,im_width,numel(gmm_fore_db));
for i = 1:numel(gmm_fore_db)
    [~,~,~,unaryU] = cluster(gmm_fore_db{i}, im_vec);
    [~,~,~,unaryB] = cluster(gmm_back_db{i}, im_vec);
    R(:,:,i) = reshape(unaryU-unaryB,im_height,im_width);
end

R_vec = reshape(permute(R,[3 1 2]), [], im_height * im_width)';
gmm_R = fitgmdist(R_vec, clusters);
K = cluster(gmm_R, R_vec);

background = mode(K);
K1 = (K~=background);
background = mode(K(K~=background));
K = K1 & (K~=background);
K = K*6 -5;
% c = hsv(numel(gmm_fore_db));
% for i=1:clusters
%     img_temp(K==i,:) = repmat(c(i,:)*255,nnz(K==i),1);
% end
out_img = reshape(K,im_height, im_width,[]);
%%
% imshow(out_img,[]);
% figure, 
imshow(img_in)
hold on;

out_img_t = out_img;
for box=120:-2:50
    bbox = fspecial('gaussian', box, 9);
    a = imfilter(out_img_t, bbox);
    
    max_response = max(max(a));
    [row,col] = find(a == max_response);
    if (numel(row)==0 || max_response<(0.99-(120-box)/500))
       continue; 
    end
    row = row(1);
    col = col(1);
    rectangle('Position',[col-box/2, row-box/2, box, box],...
         'LineWidth',2, ...
         'EdgeColor','r');
    out_img_t(max(row-box/2-5,1):min(row+box/2+5,im_height),max(col-box-5/2,1):min(col+box+5/2,im_width)) = -1;
end

for box=120:-2:25
    bbox = fspecial('gaussian', box, 8);
    a = imfilter(out_img_t, bbox);
    
    max_response = max(max(a));
    [row,col] = find(a == max_response);
    if (numel(row)==0 || max_response<(0.98-(110-box)/300))
       continue; 
    end
    row = row(1);
    col = col(1);
    rectangle('Position',[col-box/2, row-box/2, box, box],...
         'LineWidth',2, ...
         'EdgeColor','r');
    out_img_t(max(row-box/2,1):min(row+box/2,im_height),max(col-box/2,1):min(col+box/2,im_width)) = -1;
end

for box=100:-2:20
    bbox = fspecial('gaussian', box, 4);
    a = imfilter(out_img_t, bbox);
    
    max_response = max(max(a));
    [row,col] = find(a == max_response);
    if (numel(row)==0 || max_response<(0.95-(100-box)/200))
       continue; 
    end
    row = row(1);
    col = col(1);
    rectangle('Position',[col-box/2, row-box/2, box, box],...
         'LineWidth',2, ...
         'EdgeColor','r');
    out_img_t(max(row-box/2-7,1):min(row+box/2+7,im_height),max(col-box/2-7,1):min(col+box/2+7,im_width)) = -1;
end

