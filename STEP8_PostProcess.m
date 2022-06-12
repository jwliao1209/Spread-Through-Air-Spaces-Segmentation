clear; clc; close all;
addpath(genpath(fullfile('src', 'matlab')));

%% Settings
% post process for public, private (or both) prediction image
ExecuteFolder = {'public', 'private'};
Capitalize = @(str) [upper(str(1)) str(2:end)];
ImgFolder = @(exeFolder) fullfile('Dataset', [Capitalize(exeFolder) '_Image']);

% number of 50% pixel in one image
halfPixel = 0.5 * (942 * 1716);

% remove the target whose (#points < threshold).
remove_threshold = 150;
point_threshold  = 1000;

% image close+open settings
closenTimes = 1;
strelSize   = 9;

% image fill number
fillnum = 8;

% threshold settings (see Selections.m for details)
high_thres  = [0.35, 0.30, 0.20, 0.10];
lower_thres = [0.25, 0.10, 0.05, 0.01];
drop_thres  = [0.10, 0.10, 0.05, 0.01];
ratio_thres = [0.40, 0.20, 0.20, 0.20];

% detection
detect = false;

%% Post processing
for Type = ExecuteFolder
    exeFolder = Type{1};
    predLoc = fullfile('prediction', exeFolder);
    postLoc = fullfile('prediction', [exeFolder '-post']);
    
    if ~exist(postLoc, 'dir')
        mkdir(postLoc);
    end

    fileList = dir(fullfile(predLoc, '*.png'));
    assert(numel(fileList) > 0, 'There is no image in this folder');
    
    for Info = fileList'
        filename = Info.name;
        Pred = imread(fullfile(predLoc, filename)); % uint8 0-255
        Pred = double(Pred(:,:,1)) / 255;           % float 0-1
        count = 1;
        pred_fillg = Selection(Pred, lower_thres(count), high_thres(count), drop_thres(count), ratio_thres(count));
        
        if (nnz(pred_fillg) <= halfPixel)
            flag = true;
            while flag
                fprintf('stage-%d, %s\n', count, filename);
                if count > 1
                    pred_fillg = Selection(Pred, lower_thres(count), high_thres(count), drop_thres(count), ratio_thres(count));
                end
                pred = imclosen(pred_fillg, strelSize, closenTimes);

                % fill image
                pred_fillg = imagefill(pred, fillnum);
                target = dropcomponent(pred_fillg, remove_threshold);
                for i = 1 : length(target)
                    idx = target{1,i};
                    pred_fillg(idx) = 0;
                end

                flag = nnz(pred_fillg) < point_threshold;
                count = count + 1; 

                if ((count > numel(lower_thres)) && (detect == 1))
                    % use original image by sobel edge find the goal
                    pred_fillg = detection(img_loc, filename);
                    break
                end
            end
        else
            pred_fillg = zeros(size(pred_fillg));
            if detect
                pred_fillg = detection(ImgFolder(exeFolder), filename);
            end
        end
        
        imwrite(pred_fillg, fullfile(postLoc, filename));
        fprintf('Done fill hole and drop: %s.\n', filename);
    end
    fprintf(['Finished ! ' exeFolder '\n']);
end