
close all;
h = figure;
for k = 20:20
	% Create a mat filename, and load it into a structure called matData.
    fileName = sprintf('%d_dive5_2014-09-29.jpg', k);

    img = imread(fileName);
    img = imresize(img, 0.2);

    % cut the outer boundary
    img = img(140:640,200:940,:);

    features = computeImageFeatures(img);
%     features = ComputePositionColorFeatures(features);
    features = NormalizeFeatures(features);
    X = reshape(features, [], size(features, 3));
    
    for num_clusters = 3:2:15
        [idx,c] = KMeansClustering(X,num_clusters);

        height = size(img, 1);
        width = size(img, 2);

        points = zeros(height, width, 2);
        [points(:,:,1), points(:,:,2)] = meshgrid(1:width, height:-1:1);
        points = reshape(points, [], 2);

        VisualizeClusters2D(points, idx, c, h);
        fileName = sprintf('results/kmn%d_clusters%d.jpg', k, num_clusters);
        saveas(h,fileName);
    end
end

for k = 20:20
	% Create a mat filename, and load it into a structure called matData.
    fileName = sprintf('%d_dive5_2014-09-29.jpg', k);

    img = imread(fileName);
    img = imresize(img, 0.2);

    % cut the outer boundary
    img = img(140:640,200:940,:);

    features = computeImageFeatures(img);
    features = ComputePositionColorFeatures(features);
    features = NormalizeFeatures(features);
    X = reshape(features, [], size(features, 3));
    
    for num_clusters = 3:2:15
        [idx,c] = KMeansClustering(X,num_clusters);

        height = size(img, 1);
        width = size(img, 2);

        points = zeros(height, width, 2);
        [points(:,:,1), points(:,:,2)] = meshgrid(1:width, height:-1:1);
        points = reshape(points, [], 2);

        VisualizeClusters2D(points, idx, c, h);
        fileName = sprintf('results/kmn%d_clusters_position%d.jpg', k, num_clusters);
        saveas(h,fileName);
    end
end

for k = 20:20
	% Create a mat filename, and load it into a structure called matData.
    fileName = sprintf('%d_dive5_2014-09-29.jpg', k);

    img = imread(fileName);
    img = imresize(img, 0.05);

    % cut the outer boundary
    img = single(img(35:160,50:235,:));

    features = computeImageFeatures(img);
%     features = ComputePositionColorFeatures(features);
    features = NormalizeFeatures(features);
    X = reshape(features, [], size(features, 3));
    
    for num_clusters = 3:2:15
        [idx, c] = HAClustering(X,num_clusters);

        height = size(img, 1);
        width = size(img, 2);

        points = zeros(height, width, 2);
        [points(:,:,1), points(:,:,2)] = meshgrid(1:width, height:-1:1);
        points = reshape(points, [], 2);

        VisualizeClusters2D(points, idx, c, h);
        fileName = sprintf('results/kmn%d_clusters%d.jpg', k, num_clusters);
        saveas(h,fileName);
    end
end