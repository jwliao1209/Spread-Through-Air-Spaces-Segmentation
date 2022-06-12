clear; clc; close all;
addpath(genpath(fullfile('src', 'matlab')));

%% path setting
DatasetRoot = fullfile('Dataset');
AnnoRoot    = fullfile(DatasetRoot, 'Train_Annotations');
LabelRoot   = fullfile(DatasetRoot, 'Train_Labels');

AnnoList   = dir(fullfile(AnnoRoot, '*.json'));
if ~exist(LabelRoot, 'dir')
    mkdir(LabelRoot);
end

%% convert annotataion to png
for Info = AnnoList'
    % set up locations
    FileName = replace(Info.name, '.json', '');
    AnnoLoc  = fullfile(AnnoRoot, [FileName '.json']);
    LabelLoc = fullfile(LabelRoot, [FileName '.png']);

    % process data
    fprintf(['Processing case: ' FileName '\n']);
    Anno  = ReadJson(AnnoLoc);
    Label = false(Anno.imageHeight, Anno.imageWidth);
    for Shape = Anno.shapes'
        try
            Polygon = Shape.points;
        catch ME
            % label has different field cause Polygon is an array of cell, not struct
            Polygon = Shape{1}.points;
        end

        % Anno index starts from zero (maybe?)
        Polygon = Polygon + 1;
        Col = Polygon(:,1);
        Row = Polygon(:,2);

        STAS  = roipoly(Label, Col, Row);
        Label = or(Label, STAS);
    end
    imwrite(Label, LabelLoc);
end