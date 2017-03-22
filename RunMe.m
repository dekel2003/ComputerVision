addpath(genpath(pwd));

%% q1 - clustering - K-Means with 7 clusters, Position + Color features
% The clustering script that tested all the cases is in Clustering.m.
% This is just a small example.
img = imread('267_dive8_2014-09-29.jpg');
img = imresize(img, 0.2);
img = img(140:640,200:940,:);  % cut the outer boundary
features = computeImageFeatures(img);
features = ComputePositionColorFeatures(features);
features = NormalizeFeatures(features);
X = reshape(features, [], size(features, 3));
[idx,c] = KMeansClustering(X,7);
height = size(img, 1);
width = size(img, 2);
points = zeros(height, width, 2);
[points(:,:,1), points(:,:,2)] = meshgrid(1:width, height:-1:1);
points = reshape(points, [], 2);
VisualizeClusters2D(points, idx, c, figure);

%% q1 - clustering - HAC with 13 clusters, Position + Color features
% The clustering script that tested all the cases is in Clustering.m.
% This is just a small example.
% img = imread('267_dive8_2014-09-29.jpg');
% img = imresize(img, 0.04*2/3);
% img = img(20:82,30:122,:);  % cut the outer boundary
% features = computeImageFeatures(img);
% features = ComputePositionColorFeatures(features);
% features = NormalizeFeatures(features);
% X = reshape(features, [], size(features, 3));
% [idx,c] = KMeansClustering(X,13);
% height = size(img, 1);
% width = size(img, 2);
% points = zeros(height, width, 2);
% [points(:,:,1), points(:,:,2)] = meshgrid(1:width, height:-1:1);
% points = reshape(points, [], 2);
% VisualizeClusters2D(points, idx, c, figure);

%% grabcut
%grabcut('267_dive8_2014-09-29.jpg');

%% q3
%q3('267_dive8_2014-09-29.jpg');
%q3b('267_dive8_2014-09-29.jpg','267_dive8_2014-09-29_fluo.jpg');

%% q4
%q4('27_dive5_2014-09-29.jpg');