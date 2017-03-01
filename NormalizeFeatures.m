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
    [R,C,num_features] = size(features);
    
    meanFeatures = squeeze(mean(mean(features, 1)))';
    stdFeatures = std(reshape(features,[R*C, num_features]), [], 1);
    
    meanFeatures = reshape(repmat(meanFeatures, R*C, 1), [R C num_features]);
    stdFeatures = reshape(repmat(stdFeatures, R*C, 1), [R C num_features]);
    
    featuresNorm = (featuresNorm - meanFeatures) ./ stdFeatures;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                              %
%                                YOUR CODE HERE:                               %
%                                                                              %
%                HINT: The functions mean and std may be useful                %
%                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end