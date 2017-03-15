function weights = calc_weights(img, H, W)

weights = zeros(2*(H-1)*(W-1) + (H-1) + (W-1), 6);

R = 1:H-1;
C = 1:W;

gamma = 60;

im1 = imfilter(img, [-1 1]).^2;
im2 = imfilter(img, [-1;1]).^2;
% beta = inv(sum((sum(sum(im1(:,1:end-1,:)))/(H*(W-1)) + sum(sum(im2(1:end-1,:,:)/((H-1)*(W)))))*2));
beta = (mean(mean(im1(:,1:end-1,:)),2) + mean(mean(im2(1:end-1,:,:)),2))*2;
beta = mean(beta);
beta = inv(beta);

t1 = repmat((R'-1)*H,1,numel(C)) + repmat(C,numel(R),1);
t2 = repmat((R'  )*H,1,numel(C)) + repmat(C,numel(R),1);

horizontal_neighors_difference = img(R+1,C,:) - img(R,C,:);
horizontal_neighors_difference_vec = reshape(permute(horizontal_neighors_difference,[3 1 2]), 3, (H-1) * W)';

A = gamma * exp(-beta * sum(horizontal_neighors_difference_vec.^2, 2));


weights(1:numel(t1),:) = [t1(:) t2(:) zeros(numel(t1),1) A(:) A(:) zeros(numel(t1),1)];

next_index = numel(t1)+1;


R = 1:H;
C = 1:W-1;

t1 = repmat((R'-1)*H,1,numel(C))' + repmat(C,numel(R),1)';
t2 = repmat((R'-1)*H,1,numel(C))' + repmat(C+1,numel(R),1)';

horizontal_neighors_difference = img(R,C+1,:) - img(R,C,:);
horizontal_neighors_difference_vec = reshape(permute(horizontal_neighors_difference,[3 1 2]), 3, H * (W-1))';

A = exp(-beta * sum(horizontal_neighors_difference_vec.^2, 2));


weights(next_index:end,:) = [t1(:) t2(:) zeros(numel(t1),1) A(:) A(:) zeros(numel(t1),1)];


end
