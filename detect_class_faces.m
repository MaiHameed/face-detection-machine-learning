% Check out bestClassFaceDetection.[fig|png] for the image of the best results
% Warning, this script takes a painfully long time to run, reduce the
% scales array to speed it up. Otherwise, the bboxes and confidence values
% are stored in detectClass.mat

% Load workspace variables
load('cellSize.mat');
load('my_svm.mat');

confidences = zeros(0,1);

% Modifiable variables
topThreshold = 40; % Get the top x detections
overlapThreshold = 0.1;
confThreshold = 1;
scales = 0.01:0.01:2;

% load and show the image
imOriginal = im2single(rgb2gray(imread('class.jpg')));

% Display image and bounding boxes
close all;
figure;
imshow(imread('class.jpg'));
hold on;

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
        % too low, it's not a face
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

    plot_rectangle = [bbox(1), bbox(2); ...
        bbox(1), bbox(4); ...
        bbox(3), bbox(4); ...
        bbox(3), bbox(2); ...
        bbox(1), bbox(2)];
    plot(plot_rectangle(:,1), plot_rectangle(:,2), 'g-');

    % save         
    bboxes = [bboxes; bbox]; % Bounding boxes for current image
    confidences = [confidences; conf];
    numOfBoxes = numOfBoxes+1;
end

save('detectClass.mat','bboxes','confidences');