img = imread('LQ-20_dive5_2014-09-29.jpg');
%features = ComputePositionColorFeatures(img);
features = ComputeColorFeatures(img);
featuresNorm = NormalizeFeatures(features);
X = reshape(featuresNorm, [], size(featuresNorm, 3));

height = size(img, 1);
width = size(img, 2);
points = zeros(height, width, 2);
[points(:,:,1), points(:,:,2)] = meshgrid(1:width, 1:height);
points = reshape(points, [], 2);
[idx,c] = KMeansClustering(X,20);
VisualizeClusters2D(points, idx, c);