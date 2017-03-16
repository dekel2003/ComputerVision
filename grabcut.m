function grabcut(im_name)

close all;
% convert the pixel values to [0,1] for each R G B channel.
im_data = double(imread(im_name)) / 255;

% downsample the image
im_data = imresize(im_data,0.2);
im_data = im_data(140:640,200:940,:);
im_data_lab = rgb2lab(im_data);

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

% figure, imshow(im_data), hold on;
%     seg_edges = bwboundaries(inside==1);
%     visboundaries(seg_edges,'EnhanceVisibility', false);
% hold off;
% figure;

b_xmin = max(3*xmin - 2*xmax,1);
b_xmax = min(3*xmax - 2*xmin,im_width);
b_ymin = max(3*ymin - 2*ymax,1);
b_ymax = min(3*ymax - 2*ymin,im_height);
inside(1+b_ymin:b_ymax-1, 1+b_xmin:b_xmax-1) = inside(1+b_ymin:b_ymax-1, 1+b_xmin:b_xmax-1) + 1;

features = ComputePositionColorFeatures(im_data_lab);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% uncomment to add gabor features  %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% gaborArray = gaborFilterBank(2,3,17,17);
% gaborFeatureVector = gaborFeatures(rgb2gray(im_data),gaborArray,1 , 1);
% gabor_features = reshape(gaborFeatureVector, im_height, im_width, []);
% features = cat(3, features, gabor_features);
% features = NormalizeFeatures(features);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

im_vec = reshape(permute(features,[3 1 2]), [], im_height * im_width)';

fore_ind = inside==2;
back_ind = inside==1;

clusters = 5;

fore = im_vec(fore_ind, :);
back = im_vec(back_ind, :);

% fore_ind = fore_ind(:);
% back_ind = back_ind(:);

gmm_back = fitgmdist(back(:,3:end),clusters);
gmm_fore = fitgmdist(fore(:,3:end),clusters);

pairs = calc_weights(im_data, im_height, im_width);

E_prev = +Inf;
% while CONVERGENCE
while (true)
%     
%     UPDATE THE GAUSSIAN MIXTURE MODELS
    Kb = cluster(gmm_back, back(:,3:end));
    Ku = cluster(gmm_fore, fore(:,3:end));
    gmm_back = fitgmdist(back(:,3:end),clusters,'Start',Kb);
    gmm_fore = fitgmdist(fore(:,3:end),clusters,'Start',Ku);
%     
    [~,~,~,unaryB] = cluster(gmm_back, im_vec(:,3:end));
    [~,~,~,unaryU] = cluster(gmm_fore, im_vec(:,3:end));
    unary = [unaryU' ; unaryB'];
%     MAX-FLOW/MIN-CUT ENERGY MINIMIZATION
    graph = BK_Create(size(unary,2));
    BK_SetPairwise(graph, ((pairs)));
    BK_SetUnary(graph, unary);
    E = BK_Minimize(graph);
    glabels = BK_GetLabeling(graph);
    BK_Delete(graph);
    
    glabels = reshape(glabels, im_height, im_width);
    back_ind = back_ind | (glabels==1 & fore_ind);
    fore_ind = fore_ind & ~back_ind;
    
    fore = im_vec(fore_ind, :);
    back = im_vec(back_ind, :);
    
    imshow(im_data);
    hold on;
    seg_edges = bwboundaries(fore_ind);
    visboundaries(seg_edges,'EnhanceVisibility', false);
    hold off;
    drawnow;

%     IF THE ENERGY DOES NOT CONVERGE
    if (abs((E_prev-abs(E)))/E_prev < 1000*sqrt(eps))
        break;
    end
    E_prev = abs(E);

%     END

end
    

end