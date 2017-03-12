function grabcut(im_name)

% convert the pixel values to [0,1] for each R G B channel.
im_data = double(imread(im_name)) / 255;

% downsample the image
im_data = imresize(im_data,0.1);

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
disp('grabcut algorithm');


% INITIALIZE THE FOREGROUND & BACKGROUND GAUSSIAN MIXTURE MODEL (GMM)
% [C, R] = meshgrid(1:im_height, 1:im_width);
% [inside_the_box_indicesR, inside_the_box_indicesC] = find((C > xmin) & (C < xmax) & (R > ymin) & (R < ymax));
inside = zeros(im_height, im_width);
inside(1+ymin:ymax-1, 1+xmin:xmax-1) = 1;
inside_ind = inside==1;
inside_ind=inside_ind(:);

im_vec = zeros(im_height * im_width, 5);
features = ComputePositionColorFeatures(im_data);
im_vec = reshape(permute(features,[3 1 2]), 5, im_height * im_width)';

fore = im_vec(inside_ind, :);
back = im_vec(~inside_ind, :);


clusters = 5; % 5 Gaussians in each GMM
% Background
% [Kb,Cb] = kmeans(back(:,3:end), clusters, 'Distance', 'cityblock', 'Replicates', 5);
% gmm_back = gmdistribution(Cb, cov(back(:,3:end)));
gmm_back = gmdistribution.fit(back(:,3:end),clusters);
% gmm_back.CovarianceType = ''
% Foreground
% [Ku,Cu] = kmeans(fore(:,3:end), clusters, 'Distance', 'cityblock', 'Replicates', 5);
% gmm_fore = gmdistribution(Cu, cov(fore(:,3:end)));
gmm_fore = gmdistribution.fit(fore(:,3:end),clusters);


pairs = calc_weights(im_data, im_height, im_width);

% 
% while CONVERGENCE
while (true)
%     
%     UPDATE THE GAUSSIAN MIXTURE MODELS
%     Kb = cluster(gmm_back, im_vec(~inside_ind,3:end));
%     Ku = cluster(gmm_back, im_vec(inside_ind,3:end));
    gmm_back = gmdistribution.fit(im_vec(~inside_ind,3:end),clusters);
    gmm_fore = gmdistribution.fit(im_vec(inside_ind,3:end),clusters);
%     
    [~,~,~,unaryB] = cluster(gmm_back, im_vec(:,3:end));
    [~,~,~,unaryU] = cluster(gmm_fore, im_vec(:,3:end));
    unary = [unaryU' ; unaryB'];
%     MAX-FLOW/MIN-CUT ENERGY MINIMIZATION
    graph = BK_Create(size(unary,2));
    BK_SetPairwise(graph, ones(pairs));
    BK_SetUnary(graph, unary);
    E = BK_Minimize(graph);
    glabels = BK_GetLabeling(graph);
    BK_Delete(graph);
    inside_ind = (glabels==2) & inside_ind;
    inside_ind2 = (glabels==2);
%     IF THE ENERGY DOES NOT CONVERGE
%         
%         break;
%     
%     END
end

im_out = im_data;
im_out_1d = im_vec;
% Set background to white
im_out_1d(~inside_ind2, :) = 255;
% Assemble the 1D image back into 2D
for idx = 1:size(im_out, 2)
    im_out(:, idx, :) = im_out_1d((idx-1)*im_height+1:idx*im_height, 3:end);
end
imshow(im_out);


end