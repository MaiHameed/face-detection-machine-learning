%% Run if VLFeat is not set up
run('vlfeat-0.9.21/toolbox/vl_setup');

%% Get Features
close all;
clear;

pos_imageDir = 'cropped_training_images_faces';
pos_imageList = dir(sprintf('%s/*.jpg',pos_imageDir));
pos_nImages = length(pos_imageList);

neg_imageDir = 'cropped_training_images_notfaces';
neg_imageList = dir(sprintf('%s/*.jpg',neg_imageDir));
neg_nImages = length(neg_imageList);

% cellSize needs to be a factor of 36
cellSize = 4;
im = imread(sprintf('%s/%s',pos_imageDir,pos_imageList(1).name));
[imRows, imCols, ~] = size(im);
featSize = 31*(imRows/cellSize)*(imCols/cellSize);

pos_feats = zeros(pos_nImages,featSize);
for i=1:pos_nImages
    im = im2single(imread(sprintf('%s/%s',pos_imageDir,pos_imageList(i).name)));
    feat = vl_hog(im,cellSize);
    pos_feats(i,:) = feat(:);
%    fprintf('got feat for pos image %d/%d\n',i,pos_nImages);
%     imhog = vl_hog('render', feat);
%     subplot(1,2,1);
%     imshow(im);
%     subplot(1,2,2);
%     imshow(imhog)
%     pause;
end

neg_feats = zeros(neg_nImages,featSize);
for i=1:neg_nImages
    im = im2single(imread(sprintf('%s/%s',neg_imageDir,neg_imageList(i).name)));
    feat = vl_hog(im,cellSize);
    neg_feats(i,:) = feat(:);
%    fprintf('got feat for neg image %d/%d\n',i,neg_nImages);
%     imhog = vl_hog('render', feat);
%     subplot(1,2,1);
%     imshow(im);
%     subplot(1,2,2);
%     imshow(imhog)
%     pause;
end

save('pos_neg_feats.mat','pos_feats','neg_feats','pos_nImages','neg_nImages');