% you might want to have as many negative examples as positive examples
n_have = 0;
n_want = numel(dir('cropped_training_images_faces/*.jpg'));

imageDir = 'images_notfaces';
imageList = dir(sprintf('%s/*.jpg',imageDir));
nImages = length(imageList);

new_imageDir = 'cropped_training_images_notfaces';
[~, ~, ~] = mkdir(new_imageDir);

dim = 36;

i = 1;
while n_have < n_want
    % Go back to top of uncropped list if we reached the end
    if i == nImages 
        i = 1;
    end
    
    % Read in image from imageDir to crop
    imgPath = strcat(imageDir, '/', imageList(i+1).name);
    img = imread(imgPath);
    
    % Determine a random window to crop image
    [imgRows, imgCols, ~] = size(img);
    maxRow = imgRows-dim+1;
    maxCol = imgCols-dim+1;
    
    startRow = randi(maxRow);
    startCol = randi(maxCol);
    
    % Crop image
    croppedImage = img(startRow:startRow+dim-1, ...
                        startCol:startCol+dim-1, :);
    
    % Save image
    imgPath = strcat(new_imageDir, '/', int2str(n_have), '.jpg');
    imwrite(croppedImage, imgPath);
    
    % Increase counter of number of cropped images and iterate down the
    % list of uncropped images
    n_have = n_have + 1;
    i = i + 1;
end

%% Split training images into a training set and validation set
percentTraining = 80;
numOfTraining = floor((percentTraining/100)*n_have);

% Making the directories
facesDir = 'cropped_training_images_faces';
facesList = dir(sprintf('%s/*.jpg',facesDir));

notFacesDir = 'cropped_training_images_notfaces';
notFacesList = dir(sprintf('%s/*.jpg',notFacesDir));

[~,~,~] = mkdir(strcat(facesDir,'/validation'));
[~,~,~] = mkdir(strcat(facesDir,'/training'));
[~,~,~] = mkdir(strcat(notFacesDir,'/validation'));
[~,~,~] = mkdir(strcat(notFacesDir,'/training'));

for i = 1:numOfTraining
    % Copy faces training images
    source = strcat(facesDir,'/',facesList(i).name);
    dest = strcat(facesDir,'/training/',facesList(i).name);
    copyfile(source, dest);
    
    % Copy not faces training images
    source = strcat(notFacesDir,'/',notFacesList(i).name);
    dest = strcat(notFacesDir,'/training/',notFacesList(i).name);
    copyfile(source, dest);
end 

for i = numOfTraining+1:n_have
    % Copy faces validation images
    source = strcat(facesDir,'/',facesList(i).name);
    dest = strcat(facesDir,'/validation/',facesList(i).name);
    copyfile(source, dest);
    
    % Copy not faces validation images
    source = strcat(notFacesDir,'/',notFacesList(i).name);
    dest = strcat(notFacesDir,'/validation/',notFacesList(i).name);
    copyfile(source, dest);
end
   