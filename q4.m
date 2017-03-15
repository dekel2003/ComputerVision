close all;

im_name = '21_dive5_2014-09-29.jpg';
% convert the pixel values to [0,1] for each R G B channel.
im_data = double(imread(im_name)) / 255;

% downsample the image
im_data = imresize(im_data,0.2);
im_data = im_data(140:640,200:940,:);
im_data_lab = rgb2lab(im_data);
% display the image
imagesc(im_data);

% a bounding box initialization
disp('Draw a bounding box to specify the rough location of the foreground');
set(gca,'Units','pixels');
ginput(1);
p1=get(gca,'CurrentPoint');fr=rbbox;p2=get(gca,'CurrentPoint');
p=round([p1;p2]);
xmin=min(p(:,1));xmax=max(p(:,1));
ymin=min(p(:,2));ymax=max(p(:,2));
[im_height, im_width, channel_num] = size(im_data);
xmin = max(xmin, 1);
xmax = min(im_width, xmax);
ymin = max(ymin, 1);
ymax = min(im_height, ymax);

bbox = [xmin ymin xmax ymax];
line(bbox([1 3 3 1 1]),bbox([2 2 4 4 2]),'Color',[1 0 0],'LineWidth',1);
if channel_num ~= 3
    disp('This image does not have all the RGB channels, you do not need to work on it.');
    return;
end

% for h = 1 : im_height
%     for w = 1 : im_width
%         if (w > xmin) && (w < xmax) && (h > ymin) && (h < ymax)
%             % this pixel belongs to the initial foreground
%         else
%             % this pixel belongs to the initial background
%         end
%     end
% end

% grabcut algorithm


% INITIALIZE THE FOREGROUND & BACKGROUND GAUSSIAN MIXTURE MODEL (GMM)
% [C, R] = meshgrid(1:im_height, 1:im_width);
% [inside_the_box_indicesR, inside_the_box_indicesC] = find((C > xmin) & (C < xmax) & (R > ymin) & (R < ymax));
inside = zeros(im_height, im_width);
inside(1+ymin:ymax-1, 1+xmin:xmax-1) = 1;

b_xmin = max(3*xmin - 2*xmax,1);
b_xmax = min(3*xmax - 2*xmin,im_width);

b_ymin = max(3*ymin - 2*ymax,1);
b_ymax = min(3*ymax - 2*ymin,im_height);
inside(1+b_ymin:b_ymax-1, 1+b_xmin:b_xmax-1) = inside(1+b_ymin:b_ymax-1, 1+b_xmin:b_xmax-1) + 1;

features = ComputePositionColorFeatures(im_data_lab);

gaborArray = gaborFilterBank(3,2,39,39);
gaborFeatureVector = gaborFeatures(rgb2gray(im_data),gaborArray,1 , 1);
gabor_features = reshape(gaborFeatureVector, im_height, im_width, []);
features = cat(3, features, gabor_features);
features = NormalizeFeatures(features);

im_vec = reshape(permute(features,[3 1 2]), [], im_height * im_width)';

fore_ind = inside==2;
back_ind = inside==1;

clusters = 5;


back_ind_t = back_ind;
% h = figure;
h2 = figure;
subplot(2,3,1); imshow(im_data); title('Input Image');
subplot(2,3,2); imshow(fore_ind); title('Initialization');
for i=1:6
    figure(h2);
    fore_ind=fore_ind(:);
    back_ind=back_ind(:) & back_ind_t(:);
    
    fore = im_vec(fore_ind, :);
    back = im_vec(back_ind, :);

    gmm_fore = fitgmdist(fore(:,3:end),clusters);
    gmm_back = fitgmdist(back(:,3:end),clusters);

    [~,~,~,unaryU] = cluster(gmm_fore, im_vec(:,3:end));
    [~,~,~,unaryB] = cluster(gmm_back, im_vec(:,3:end));
    
%     unaryU(inside==2) = unaryU(inside==2) * 1.01;
%     unaryB(~inside==2) = unaryB(~inside==2) * 1.01;

    U_img = reshape(1.1*unaryU-unaryB,im_height, im_width);
%     imshow(U_img,[-10,10])
    m = fore_ind==1;
    m = reshape(m,im_height, im_width);
%     subplot(2,3,1); imshow(im_data); title('Input Image');
%     subplot(2,3,2); imshow(m); title('Initialization');
    subplot(2,3,3); imshow(U_img,[-10,10]); title('Energy Image');
    subplot(2,3,4); title('Segmentation');
    seg = region_seg(U_img, m, 100, 0.8); %-- Run segmentation
    subplot(2,3,5); imshow(seg); title('Global Region-Based Segmentation');
    img = im_data;
    
    subplot(2,3,6); imshow(img); title('Result'); 
    hold on;
    seg_edges = bwboundaries(seg);    
    visboundaries(seg_edges,'EnhanceVisibility', false);
    hold off;
    
    fore_ind = seg==1;
    back_ind_t = seg==0;
    drawnow;
    
%     figure(h);
%     if mod(i,2)==1
%         subplot(2,3,(i-1)/2+1); imshow(img); title('Result');
%     end
%     drawnow;
end
