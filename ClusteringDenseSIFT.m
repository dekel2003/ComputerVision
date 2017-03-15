I = imread('Eilat_REF_FLR_pairs/20_dive5_2014-09-29.jpg');
I = imresize(I, 0.25);
I = single(rgb2gray(I));
[F,D] = vl_dsift(I);
F = F';
D = NormalizeFeatures(D');
[idx,c] = KMeansClustering(D,8);
VisualizeClusters2D(F, idx, c);