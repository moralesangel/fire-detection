clear%% Clean Workspace
clear;
close all;
clc;

%% Parameters
model_to_use = '10-02-24-23_0.60153';
trainCNN = false; % False to load a model and true to train the CNN
indexesExamples = randperm(2003, 3); % Between 1 and 2003

%% Getting data
folder_im = 'incendios_segmentados/Resized_Images/';
folder_seg = 'incendios_segmentados/Resized_Masks/';
imds_im = imageDatastore(folder_im);
imds_seg = imageDatastore(folder_seg);

folder_labels = 'incendios_segmentados/Resized_Masks/';
classNames = ["woods", "fire"]';
pixelLabelID = {0, 1};
pxds = pixelLabelDatastore(folder_labels, classNames, pixelLabelID);

%% Splitting Data
[imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = ...
partitionData(imds_im,pxds, pixelLabelID, 1, 0.6, 0.2);

dsTrain = combine(imdsTrain, pxdsTrain);
dsVal = combine(imdsVal, pxdsVal);
dsTest = combine(imdsTest, pxdsTest);

[M, N, channels] = size(readimage(imds_im,1));

%% Convolutional Neuronal Network
if trainCNN
    disp('Training CNN...');

    fire = 0.96;
    classWeights = [1-fire;fire];
%     data = load('models/NET.mat');
%     lgraph = segnetLayers([256, 256, 3], 2, data.net);

    lgraph = [
        imageInputLayer([M, N, channels])
        convolution2dLayer(3,32,'Padding',1);
        reluLayer
        maxPooling2dLayer(2,'Stride',2)
        transposedConv2dLayer(4,32,'Stride',2,'Cropping',1)
        reluLayer
        convolution2dLayer(1,2);
        softmaxLayer
        pixelClassificationLayer('Name', 'pixelLabels', ...
                                 'ClassNames', classNames, ...
                                 'ClassWeights', classWeights);
    ];
    
    % Stochastic Gradient Descent with Momentum
    options = trainingOptions('sgdm', ...
     'InitialLearnRate',0.001, ...
     'MiniBatchSize',10, ...
     'MaxEpochs', 2, ...
     'ValidationData', dsVal, ...
     'ValidationFrequency',20, ...
     'Shuffle','every-epoch', ...
     'Verbose',true, ...
     'Plots','training-progress');
    net = trainNetwork(dsTrain,lgraph, options);
    net2 = train;
else
    disp('Loading model...');
    load(strcat('models/', model_to_use),'net');
end

%% Confusion Matrix
disp('Showing results...');
numImages = numel(imdsTest.Files);
batchSize = 100;
confMat = zeros(numel(classNames));

for i = 1:batchSize:batchSize
    endIndex = min(i + batchSize - 1, numImages);
    imdsBatch = subset(imdsTest, i:endIndex);
    pxdsBatch = subset(pxdsTest, i:endIndex);
    
    pxdsResultsBatch = semanticseg(imdsBatch, net, "WriteLocation", tempdir);
    metricsBatch = evaluateSemanticSegmentation(pxdsResultsBatch, pxdsBatch);
    
    confMat = confMat + table2array(metricsBatch.ConfusionMatrix);
end
figure;confusionchart(confMat);

%% IoU (Intersection over Union)
numClasses = size(confMat, 1);
IoU = zeros(numClasses, 1);
for i = 1:numClasses
    intersection = confMat(i,i);
    union = sum(confMat(i,:)) + sum(confMat(:,i)) - confMat(i,i);
    IoU(i) = intersection / union;
end
meanIoU = mean(IoU);

%% Save model
currentDateTime = datetime('now');
formattedDate = datestr(currentDateTime, 'dd-HH-MM-SS');
model_name = [formattedDate '_' num2str(meanIoU) '.mat'];

if trainCNN
    disp('Saving model...');
    save(strcat('models/', model_name),'net');
end

%% Best 3 Examples
disp('Finding best examples...');
% IoU for each image
numImages = numel(imdsTest.Files);
IoUs = zeros(numImages, 1);
classIndex = 2; % Índice de la segunda clas

for i = 1:numImages
    % Realizar la segmentación semántica
    img = readimage(imds_im, i);
    C = semanticseg(img, net);

    % Obtener las etiquetas verdaderas y predichas como vectores
    groundTruth = readimage(pxdsTest, i);
    groundTruthVector = groundTruth(:);   % Convertir la matriz en vector
    predictedLabelsVector = C(:); 
    confMat = confusionmat(groundTruthVector, predictedLabelsVector);

%     % Calcular el IoU para esta imagen
%     intersection = diag(confMat);
%     union = sum(confMat, 2) + sum(confMat, 1)' - intersection;
%     IoU = intersection ./ union;
%     IoUs(i) = mean(IoU, 'omitnan');  % promedio de IoU para todas las clases

    % Calcular el IoU solo para la segunda clase
    intersection = confMat(classIndex, classIndex);
    union = sum(confMat(classIndex, :)) + sum(confMat(:, classIndex)) - intersection;
    IoU = intersection / union;
    IoUs(i) = IoU;  % IoU para la segunda clase

end

% Greatest IoUs
[sortedIoUs, sortedIndices] = sort(IoUs, 'descend');
top3Indices = sortedIndices(1:3);

classColors = [0 1 1; 1 0 0];
figure;
for i=1:size(top3Indices, 1)
    C = semanticseg(readimage(imds_im,indexesExamples(i)),net);
    B = labeloverlay(readimage(imds_im,indexesExamples(i)), C, 'Colormap', classColors, 'Transparency',0.4);
    real_img = readimage(imds_im, indexesExamples(i));
    
    % Show 
    subplot(length(indexesExamples), 3, (i-1)*3+1);imshow(real_img);
    subplot(length(indexesExamples), 3, (i-1)*3+2);imshow(readimage(imds_seg,indexesExamples(i))*255);
    subplot(length(indexesExamples), 3, (i-1)*3+3);imshow(B);
    
end

%% Random Example
p = randperm(2003, 1);
classColors = [0 1 1; 1 0 0];

C = semanticseg(readimage(imds_im,p),net);
B = labeloverlay(readimage(imds_im,p), C, 'Colormap', classColors, 'Transparency',0.4);

figure;
subplot(1, 3, 1);imshow(readimage(imds_im, p));
subplot(1, 3, 2);imshow(readimage(imds_seg, p)*255);
subplot(1, 3, 3);imshow(B);

