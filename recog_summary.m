disp('Please read the comments in recog_summary.m since writing paragraphs in disp() is painful');

% To generate the non-faces we took the large non-faces image and randomly sampled a 36x36 pixel area
% then repeated it down the list of images, wrapping back up to the top of the list and down again
% until we got the total number of images needed. 
% The get_features.m script was minimally modified, with the functionality unchanged. 
% The train_svm.m file was where we split our data into training and validation sets, as well as 
% trained and determine classifier performance. 

% The best classifier performance is: 
% Classifier performance on validation data: 
% accuracy:   0.977 
% true  positive rate: 0.479 
% false positive rate: 0.002 
% true  negative rate: 0.498 
% false negative rate: 0.021 

% We used trial and error to determine the best performance 


