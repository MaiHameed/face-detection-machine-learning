disp('Please read the comments in detect_summary.m since writing paragraphs in disp() is painful');

% The class face detection can be found in detectClass.m
% We scaled the image down and up, and calculated all the bounding boxes
% at each scale above a tiny confidence threshold
% Once the list of all bboxes were found, we sorted them in descending order
% with respect to the confidence measurement
% Starting from the most confident box, and going down the list, we stored the
% most confident box and discarded any overlaps which passed a certain
% overlap ratio

% We also found a data set of faces from this
% linkhttps://github.com/NVlabs/ffhq-dataset dataset that we used to further train
% our data. Surprisingly, it made our classifier perform worse, from an
% average precision of 0.73 to 0.5. The additional faces were saved in the
% cropped_training_images_faces directory

% average_precision.png is the precision gained from the original dataset.
% average_precision_bigger_dataset.png is the precision gained from the
% expanded dataset of 15k+ images.