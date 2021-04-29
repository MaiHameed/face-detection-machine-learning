%% Run if VLFeat is not set up
run('vlfeat-0.9.20/toolbox/vl_setup');

%% Split into training and validation
close all;
clear;

load('pos_neg_feats.mat');

percentTraining = 80;
numOfTraining = floor((percentTraining/100)*pos_nImages);

trainFeats = cat(1,pos_feats(1:numOfTraining,:), ...
                    neg_feats(1:numOfTraining,:));
validFeats = cat(1,pos_feats(1+numOfTraining:end,:), ...
                    neg_feats(1+numOfTraining:end,:));
trainLabels = cat(1,ones(numOfTraining,1), ...
                    -1*ones(numOfTraining,1));
validLabels = cat(1,ones(pos_nImages-numOfTraining,1), ...
                    -1*ones(pos_nImages-numOfTraining,1));
             
%% Train

lambda = 0.0645;
[w,b] = vl_svmtrain(trainFeats',trainLabels',lambda);

fprintf('Classifier performance on train data:\n')
confidences = [pos_feats(1:numOfTraining,:); neg_feats(1:numOfTraining,:)]*w + b;
[tp_rate, fp_rate, tn_rate, fn_rate] =  report_accuracy(confidences, trainLabels);

fprintf('Classifier performance on validation data:\n')
confidences = [pos_feats(1+numOfTraining:end,:); neg_feats(1+numOfTraining:end,:)]*w + b;
[tp_rate, fp_rate, tn_rate, fn_rate] =  report_accuracy(confidences, validLabels);

save('my_svm.mat','w','b');