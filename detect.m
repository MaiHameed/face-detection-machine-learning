%% Import VLFeat
run('vlfeat-0.9.21/toolbox/vl_setup');

%% Detect
clear;
imageDir = 'test_images';
imageList = dir(sprintf('%s/*.jpg',imageDir));
nImages = length(imageList);

% Load workspace variables
load('cellSize.mat');
load('my_svm.mat');

%bboxes = zeros(0,4);
confidences = zeros(0,1);
image_names = cell(0,1);

bboxesTotal = [];

dim = 36;
for i=1:nImages
    % load and show the image
    im = im2single(imread(sprintf('%s/%s',imageDir,imageList(i).name)));
    
    %close all;
    %figure;
    %imshow(im);
    %hold on;
    
    % generate a grid of features across the entire image. you may want to 
    % try generating features more densely (i.e., not in a grid)
    feats = vl_hog(im,cellSize);
    
    % concatenate the features into 6x6 bins, and classify them (as if they
    % represent 36x36-pixel faces)
    [rows,cols,~] = size(feats); 
    confs = zeros(rows,cols);
    for r=1:rows-cellSize+1
        for c=1:cols-cellSize+1
        % create feature vector for the current window and classify it using the SVM model, 
        % take dot product between feature vector and w and add b,
	    % store the result in the matrix of confidence scores confs(r,c)
        window = feats(r:r+cellSize-1,c:c+cellSize-1,:);
        confs(r,c) = w'*window(:)+b;
        end
    end
       
    % get the most confident predictions 
    [~,inds] = sort(confs(:),'descend');
    bboxes = []; % Clear bounding boxes per image
    %inds = inds(1:40); % (use a bigger number for better recall)
    for n=1:numel(inds)        
        [row,col] = ind2sub([size(feats,1) size(feats,2)],inds(n));
        
        bbox = [ col*cellSize ...
                 row*cellSize ...
                (col+cellSize-1)*cellSize ...
                (row+cellSize-1)*cellSize];
        conf = confs(row,col);
        image_name = {imageList(i).name};
        
        %plot_rectangle = [bbox(1), bbox(2); ...
        %    bbox(1), bbox(4); ...
        %    bbox(3), bbox(4); ...
        %    bbox(3), bbox(2); ...
        %    bbox(1), bbox(2)];
        %plot(plot_rectangle(:,1), plot_rectangle(:,2), 'g-');
        
        % save         
        bboxes = [bboxes; bbox]; % Bounding boxes for current image
        confidences = [confidences; conf];
        image_names = [image_names; image_name];
    end
    
    topThreshold = 20; % Get the top x detections
    overlapThreshold = 0.5;
    bboxesNMS = [];
    confidencesNMS = [];
    image_namesNMS = [];
    for j=1:topThreshold
        if(inds(j) == 0)
            continue;
        end
        % get bbox coordinates of top detections
        bb = bboxes(inds(j),:); 
        bboxesNMS = [bboxesNMS; bb];
        confidencesNMS = confidences(inds(j));
        image_namesNMS = image_names(inds(j));
        
        % Non Max Suppression 
        % (compare top confidence box to all other boxes and 
        % remove overlaps) 
        for n=j+1:numel(inds)
            % Ensure we don't compare with a previously detected overlap
            % or compare the same two boxes
            if(inds(n) == 0)
                continue;
            end
            bb2 = bboxes(inds(n),:);
            % Intersect box
            bi=[max(bb(1),bb2(1)); 
                max(bb(2),bb2(2)); 
                min(bb(3),bb2(3)); 
                min(bb(4),bb2(4))];
            % Width and height of intersect
            iw=bi(3)-bi(1)+1;
            ih=bi(4)-bi(2)+1;
            if iw>0 && ih>0 % Overlap detected!      
                % compute overlap as area of intersection / area of union
                ua=(bb(3)-bb(1)+1)*(bb(4)-bb(2)+1)+...
                   (bb2(3)-bb2(1)+1)*(bb2(4)-bb2(2)+1)-...
                   iw*ih;
                ov=iw*ih/ua;
                if ov>overlapThreshold
                    % Delete overlapping box
                    inds(j) = 0;
                end
            end
        end
    end
    bboxesTotal = [bboxesTotal; bboxesNMS];
    
    % Display image and bounding boxes
    close all;
    figure;
    imshow(im);
    hold on;
    for m=1:size(bboxesNMS,1)
        bbox = bboxesNMS(m,:);
        plot_rectangle = [bbox(1), bbox(2); ...
            bbox(1), bbox(4); ...
            bbox(3), bbox(4); ...
            bbox(3), bbox(2); ...
            bbox(1), bbox(2)];
        plot(plot_rectangle(:,1), plot_rectangle(:,2), 'g-');
    end    
    
    fprintf('got preds for image %d/%d\n', i,nImages);
    pause;
end

% evaluate
label_path = 'test_images_gt.txt';
[gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections] = ...
    evaluate_detections_on_test(bboxes, confidences, image_names, label_path);
