function featuresNorm = NormalizeFeatures(features)
% Normalize image features to have zero mean and unit variance. This
% normalization can cause k-means clustering to perform better.
%
% INPUTS
% features - An array of features for an image. features(i, j, :) is the
%            feature vector for the pixel img(i, j, :) of the original
%            image.
%
% OUTPUTS
% featuresNorm - An array of the same shape as features where each feature
%                has been normalized to have zero mean and unit variance.

    features = double(features);
    featuresNorm = features;
    [r, c, f] = size(features);
    F = reshape(features, [], 1, f);
    m = mean(F);
    s = std(F);
    T = repmat(m, r, c, 1);
    featuresNorm = featuresNorm - T;
    T = repmat(s, r, c, 1);
    featuresNorm = featuresNorm ./ T;
end