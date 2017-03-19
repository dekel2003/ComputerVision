function [ featureList ] = computeImageFeatures( img )
% Implementation for question1, section c.

[Rows,Cols,~] = size(img);

%% take LAB colors:
labImg = rgb2lab(img);
featureList = labImg;

%% gabor filter:
% for colorChannel = 1:3
%     gaborArray = gaborFilterBank(5,8,39,39);
%     gaborFeatureVector = gaborFeatures(labImg(:,:,colorChannel),gaborArray,1 , 1);
%     features = reshape(gaborFeatureVector, Rows, Cols, []);
%     featureList = cat(3, featureList, features);
% end

end

