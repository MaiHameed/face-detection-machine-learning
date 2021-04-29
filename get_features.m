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
cellSize = 6;
im = imread(sprintf('%s/%s',pos_imageDir,pos_imageList(1).name));
[imRows, imCols, ~] = size(im);
featSize = 31*(imRows/cellSize)*(imCols/cellSize);

pos_feats = zeros(pos_nImages*2,featSize);
n = 1;
for i=1:2:pos_nImages*2
    im = im2single(imread(sprintf('%s/%s',pos_imageDir,pos_imageList(n).name)));
    imFlip = flip(im,2);
    
    feat = vl_hog(im,cellSize);
    featFlip = vl_hog(imFlip,cellSize);
    
    pos_feats(i,:) = feat(:);
    pos_feats(i+1,:) = featFlip(:);
    
    n = n + 1;
%    fprintf('got feat for pos image %d/%d\n',i,pos_nImages);
%     imhog = vl_hog('render', feat);
%     subplot(1,2,1);
%     imshow(im);
%     subplot(1,2,2);
%     imshow(imhog)
%     pause;
end

neg_feats = zeros(neg_nImages*2,featSize);
n = 1;
for i=1:2:neg_nImages*2
    im = im2single(imread(sprintf('%s/%s',neg_imageDir,neg_imageList(n).name)));
    imFlip = flip(im,2);
    
    feat = vl_hog(im,cellSize);
    featFlip = vl_hog(imFlip,cellSize);
    
    neg_feats(i,:) = feat(:);
    neg_feats(i+1,:) = featFlip(:);
    n = n+1;
%    fprintf('got feat for neg image %d/%d\n',i,neg_nImages);
%     imhog = vl_hog('render', feat);
%     subplot(1,2,1);
%     imshow(im);
%     subplot(1,2,2);
%     imshow(imhog)
%     pause;
end

pos_nImages = pos_nImages*2;
neg_nImages = neg_nImages*2;

save('pos_neg_feats.mat','pos_feats','neg_feats','pos_nImages','neg_nImages');
save('cellSize.mat','cellSize');