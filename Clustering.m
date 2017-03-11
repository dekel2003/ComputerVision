img = imread('Eilat_REF_FLR_pairs/20_dive5_2014-09-29.jpg');
img = imresize(img, 0.25);

features = computeImageFeatures(img);
featuresNorm = NormalizeFeatures(features);
X = reshape(featuresNorm, [], size(featuresNorm, 3));  

[idx,c] = KMeansClustering(X,8);

height = size(img, 1);
width = size(img, 2);
points = zeros(height, width, 2);
[points(:,:,1), points(:,:,2)] = meshgrid(1:width, 1:height);
points = reshape(points, [], 2);

VisualizeClusters2D(points, idx, c);