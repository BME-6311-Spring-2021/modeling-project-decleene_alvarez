net=googlenet;
imds = imageDatastore("Skin cancer","IncludeSubfolders",true,"LabelSource","foldernames");
[trainImgs,testImgs] = splitEachLabel(imds,0.8);
[trainImgs, valImgs]=splitEachLabel(trainImgs,0.5);
Train1 = augmentedImageDatastore([224 224],trainImgs);
Test1 = augmentedImageDatastore([224 224],testImgs);
Val1 = augmentedImageDatastore([224 224],valImgs);
numClasses = numel(categories(imds.Labels));
skinLabels = imds.Labels;
lg =layerGraph(net);
fc = fullyConnectedLayer(9,"Name","new_layer");
lg = replaceLayer(lg,"loss3-classifier",fc);
cl = classificationLayer("Name","Lout");
lg = replaceLayer(lg,"output",cl);
options = trainingOptions("sgdm","Plots","training-progress","Momentum",0.95);
%%
[skinnet,info] = trainNetwork(Train1, lg, options);

%%
testp = classify(skinnet,Val1);
nnz(testp == valImgs.Labels)/numel(testp)
%%
options1 =trainingOptions("sgdm","InitialLearnRate",0.001,"Plots","training-progress");
[skinnet2, info] = trainNetwork(Train1, lg, options1)
%%
testp1 = classify(skinnet2,Val1);
nnz(testp1 == valImgs.Labels)/numel(testp1)
%%
testp2 = classify(skinnet2,Test1);
nnz(testp2 == testImgs.Labels)/numel(testp2)
confusionchart(testImgs.Labels,testp2)