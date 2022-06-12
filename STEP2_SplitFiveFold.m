clear; clc; close all;
addpath(genpath(fullfile('src', 'matlab')));

%% setup locations
DatasetRoot = fullfile('Dataset');

AnnoRoot    = fullfile(DatasetRoot, 'Train_Annotations');
LabelRoot   = fullfile(DatasetRoot, 'Train_Labels');
FileList    = dir(fullfile(AnnoRoot, '*.json'));

JsonRoot    = fullfile('Json_Data');
if ~exist(JsonRoot, 'dir')
    mkdir(JsonRoot);
end

ratioLB = [0.00 0.01 0.02 0.03 0.04 0.05 0.10];
ratioUB = [0.01 0.02 0.03 0.04 0.05 0.10 1.00];
testNum = [  68   32   12    7    4    7    3];

%% Loop over the data
% loop backward such that preallocate memory in the first iteration
for k = numel(FileList): -1 : 1
    Info = FileList(k);
    FileName = replace(Info.name, '.json', '');
    fprintf(['Processing ' FileName '\n']);
    Anno  = ReadJson(fullfile(AnnoRoot, [FileName '.json']));
    Label = imread(fullfile(LabelRoot, [FileName '.png']));

    Stat(k,1) = SplitFoldParse(FileName, Anno, Label);
end

%% Sort by STAS ratio and split into five categories
Stat = struct2table(Stat);
Stat = sortrows(Stat, 'ratio');

TrainSet = cell(numel(ratioLB), 1);
TestSet  = [];
ExtraSet = [];
for k = numel(ratioLB) : -1 : 1
    Begin = find(Stat.ratio >= ratioLB(k), 1, 'first');
    End   = find(Stat.ratio <  ratioUB(k), 1, 'last');
    Temp  = table2struct(Stat(Begin:End,:));
    TempNum = size(Temp, 1);

    SortIdx  = randperm(TempNum);
    Temp     = Temp(SortIdx,:);
    TestIdx  = testNum(k);
    ExtraIdx = TestIdx + mod(TempNum-TestIdx, 5);
    
    TrainSet{k} = Temp(ExtraIdx+1:end, :);
    try
        TestSet  = cat(1, TestSet,  Temp(1:TestIdx, :));
        ExtraSet = cat(1, ExtraSet, Temp(TestIdx+1:ExtraIdx,:));
    catch ME
        TestSet  = Temp(1:TestIdx, :);
        ExtraSet = Temp(TestIdx+1:ExtraIdx,:);
    end
end

%% Write json files
for fold = 0 : 4
    TrainContent = [];
    ValidContent = [];

    for CellData = TrainSet'
        StructData = CellData{1};
        DataNum = numel(StructData);
        
        Begin = (fold)  * (DataNum/5) + 1;
        End   = (fold+1)* (DataNum/5);
        
        ValidID = Begin:End;
        TrainID = setdiff(1:DataNum, ValidID);
        try
            TrainContent = cat(1, TrainContent, StructData(TrainID));
            ValidContent = cat(1, ValidContent, StructData(ValidID));
        catch ME
            TrainContent = StructData(TrainID);
            ValidContent = StructData(ValidID);
        end
    end

    WriteJson(fullfile(JsonRoot, ['Fold' num2str(fold) '_train.json']), ...
              TrainContent);
    WriteJson(fullfile(JsonRoot, ['Fold' num2str(fold) '_valid.json']), ...
              ValidContent);
end

WriteJson(fullfile(JsonRoot, 'Test.json'), TestSet);
