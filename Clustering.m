
close all;
h = figure;

%% K-Means, color
for k = 20:29
	% Create a mat filename, and load it into a structure called matData.
    fileName = sprintf('%d_dive5_2014-09-29.jpg', k);

    img = imread(fileName);
    img = imresize(img, 0.2);

    % cut the outer boundary
    img = img(140:640,200:940,:);

    features = computeImageFeatures(img);
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

%% K-Means, color + position
for k = 20:29
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

%% K-Means, gabor filters
%%%% add gabor:
gaborArray = gaborFilterBank(2,3,17,17);

for k = 20:29
	% Create a mat filename, and load it into a structure called matData.
    fileName = sprintf('%d_dive5_2014-09-29.jpg', k);

    img = imread(fileName);
    img = imresize(img, 0.2);

    % cut the outer boundary
    img = img(140:640,200:940,:);
    
    height = size(img, 1);
    width = size(img, 2);

    features = computeImageFeatures(img);
    gaborFeatureVector = gaborFeatures(rgb2gray(img),gaborArray,1 , 1);
    gabor_features = reshape(gaborFeatureVector, height, width, []);
    features = cat(3, features, gabor_features);
    features = NormalizeFeatures(features);
    X = reshape(features, [], size(features, 3));
    
    for num_clusters = 3:2:15
        [idx,c] = KMeansClustering(X,num_clusters);

        points = zeros(height, width, 2);
        [points(:,:,1), points(:,:,2)] = meshgrid(1:width, height:-1:1);
        points = reshape(points, [], 2);

        VisualizeClusters2D(points, idx, c, h);
        fileName = sprintf('results/kmn%d_clusters_gabor%d.jpg', k, num_clusters);
        saveas(h,fileName);
    end
end


%% HAC, different kinds are in inner loops
for k = 20:29
	% Create a mat filename, and load it into a structure called matData.
    fileName = sprintf('%d_dive5_2014-09-29.jpg', k);

    img = imread(fileName);
    img = imresize(img, 0.04*2/3);

    % cut the outer boundary
    img = img(20:82,30:122,:);
%     imwrite(img,sprintf('results/hac%d_raw.jpg', k));

    features = computeImageFeatures(img);
%     features = ComputePositionColorFeatures(features);
    features = NormalizeFeatures(features);
    X = single(reshape(features, [], size(features, 3)));
    
    height = size(img, 1);
    width = size(img, 2);
        
    for num_clusters = 13:8:29
        [idx, c] = HAClustering(X,num_clusters);



        points = zeros(height, width, 2);
        [points(:,:,1), points(:,:,2)] = meshgrid(1:width, height:-1:1);
        points = reshape(points, [], 2);

        VisualizeClusters2D(points, idx, c, h);
        fileName = sprintf('results/hac%d_clusters%d.jpg', k, num_clusters);
        saveas(h,fileName);
    end
    
    features = computeImageFeatures(img);
    features = ComputePositionColorFeatures(features);
    features = NormalizeFeatures(features);
    X = single(reshape(features, [], size(features, 3)));
    
    for num_clusters = 13:8:29
        [idx, c] = HAClustering(X,num_clusters);

        points = zeros(height, width, 2);
        [points(:,:,1), points(:,:,2)] = meshgrid(1:width, height:-1:1);
        points = reshape(points, [], 2);

        VisualizeClusters2D(points, idx, c, h);
        fileName = sprintf('results/hac%d_clusters%d_position.jpg', k, num_clusters);
        saveas(h,fileName);
    end
    
    features = computeImageFeatures(img);
    gaborFeatureVector = gaborFeatures(rgb2gray(img),gaborArray,1 , 1);
    gabor_features = reshape(gaborFeatureVector, height, width, []);
    features = cat(3, features, gabor_features);
    features = NormalizeFeatures(features);
    X = reshape(features, [], size(features, 3));
    
    for num_clusters = 13:8:29
        [idx, c] = HAClustering(X,num_clusters);

        points = zeros(height, width, 2);
        [points(:,:,1), points(:,:,2)] = meshgrid(1:width, height:-1:1);
        points = reshape(points, [], 2);

        VisualizeClusters2D(points, idx, c, h);
        fileName = sprintf('results/hac%d_clusters%d_gabor.jpg', k, num_clusters);
        saveas(h,fileName);
    end
    
end