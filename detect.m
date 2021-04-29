%% Import VLFeat
run('vlfeat-0.9.21/toolbox/vl_setup');

%% Detect

% Warning, this script takes quite a long time to run, read through
% detect_summary.m for an overview and precision results.

clear;
imageDir = 'test_images';
imageList = dir(sprintf('%s/*.jpg',imageDir));
nImages = length(imageList);

% Load workspace variables
load('cellSize.mat');
load('my_svm.mat');

bboxesTotal = zeros(0,4);
confidences = zeros(0,1);
image_names = cell(0,1);

bboxesTotal = [];

% Modifiable variables
topThreshold = 30; % Get the top x detections
overlapThreshold = 0.05;
confThreshold = 1;
shouldDisplay = 0;
scales = 0.02:0.02:1.6;

dim = 36;
for i=1:nImages
    % load and show the image
    imOriginal = im2single(imread(sprintf('%s/%s',imageDir,imageList(i).name)));
    image_name = {imageList(i).name};
    
    if(shouldDisplay)
        % Display image and bounding boxes
        close all;
        figure;
        imshow(imOriginal);
        hold on;
    end
    
    % Store bbox coordinate info relative to original image from all
    % scalings
    bboxesFromScales = [];
    % Store confidence info
    confsFromScales = [];
    
    for k=1:numel(scales)
        
        % Scale image
        scale = scales(k);
        im = imresize(imOriginal, scale);                

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
            conf = w'*window(:)+b;
            
            % Don't even bother with this bounding box if the confidence is
            % too low
            if (conf < confThreshold)
                continue;
            end
            
            % Calculate bbox relative to original scaled image
            bbox = [ (c*cellSize)/scale ...
                     (r*cellSize)/scale ...
                    ((c+cellSize-1)*cellSize)/scale ...
                    ((r+cellSize-1)*cellSize)/scale];
            bboxesFromScales = [bboxesFromScales; bbox]; % Append bbox to total boxes
            confsFromScales = [confsFromScales; conf];
            end
        end
    end
    
    % Non-maximum suppression
    [~, inds] = sort(confsFromScales(:),'descend');
    indexOfBbox = 1;
    numOfBoxes = 0;
    bboxes = [];
    while(numOfBoxes < topThreshold)
        % Break out of loop if we ran through all possible bounding boxes
        if(indexOfBbox > size(inds,1))
            break;
        end
        
        % Get current bounding box
        bbox = bboxesFromScales(inds(indexOfBbox),:);
        conf = confsFromScales(inds(indexOfBbox));
        shouldSave = 1;
        
        % For the currently selected bounding box, check if it overlaps
        % with any of the already saved boxes. If yes, and ratio is
        % over a threshold, discard.
        for j=1:numOfBoxes
            if(numOfBoxes == 0)
                break;
            end
            
            % bbox2 iterates through all already saved boxes
            bbox2 = bboxes(j,:);
            
            % Intersect box
            bi=[max(bbox(1),bbox2(1)); 
                max(bbox(2),bbox2(2)); 
                min(bbox(3),bbox2(3)); 
                min(bbox(4),bbox2(4))];
            
            % Width and height of intersect
            iw=bi(3)-bi(1)+1;
            ih=bi(4)-bi(2)+1;
            
            if iw>0 && ih>0 % Overlap detected!     
                % compute overlap as area of intersection / area of union
                ua=(bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1)+...
                   (bbox2(3)-bbox2(1)+1)*(bbox2(4)-bbox2(2)+1)-...
                   iw*ih;
                ov=iw*ih/ua;
                if ov>overlapThreshold
                    % Delete overlapping box
                    shouldSave = 0;
                end
            end
        end
        
        % Skip saving current box details if overlap was detected
        indexOfBbox = indexOfBbox+1;
        if(~shouldSave)
            continue
        end
        
        if(shouldDisplay)
            plot_rectangle = [bbox(1), bbox(2); ...
                bbox(1), bbox(4); ...
                bbox(3), bbox(4); ...
                bbox(3), bbox(2); ...
                bbox(1), bbox(2)];
            plot(plot_rectangle(:,1), plot_rectangle(:,2), 'g-');
        end
        
        % save         
        bboxes = [bboxes; bbox]; % Bounding boxes for current image
        confidences = [confidences; conf];
        image_names = [image_names; image_name];
        numOfBoxes = numOfBoxes+1;
    end
    
    bboxesTotal = [bboxesTotal; bboxes];
    if(shouldDisplay)
        fprintf('got preds for image %d/%d\n', i,nImages);
        pause;
    end
end

% evaluate
label_path = 'test_images_gt.txt';
[gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections] = ...
    evaluate_detections_on_test(bboxesTotal, confidences, image_names, label_path);
