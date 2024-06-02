function [imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] ...
    = partitionData(imds_im,pxds, pixelLabelID, dataFraction, trainP, valP)

    if nargin < 3
        dataFraction = 1;
    end

    % Calcula el número total de archivos y selecciona un subconjunto
    totalNumFiles = numel(imds_im.Files);
    numFilesToUse = round(dataFraction * totalNumFiles);
    idx = randperm(totalNumFiles, numFilesToUse);
    
    % Dividir en entrenamiento, validación y prueba
    numTrain = round(trainP * numFilesToUse);
    numVal = round(valP * numFilesToUse);
    
    idxTrain = idx(1:numTrain);
    idxVal = idx(numTrain+1:numTrain+numVal);
    idxTest = idx(numTrain+numVal+1:end);
    
    imdsTrain = subset(imds_im, idxTrain);
    pxdsTrain = subset(pxds, idxTrain);
    
    imdsVal = subset(imds_im, idxVal);
    pxdsVal = subset(pxds, idxVal);
    
    imdsTest = subset(imds_im, idxTest);
    pxdsTest = subset(pxds, idxTest);

end

