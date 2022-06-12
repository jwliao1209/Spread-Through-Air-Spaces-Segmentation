function IsEqual = Compare(ImgLoc1, ImgLoc2, Suffix)
    % Check whether the prediction (or more general, image)
    % are identical from two folders.
    % These two folders must contains same filename's image.
    if ~exist('Suffix', 'var')
        Suffix = 'png';
    end
    IsEqual  = true;
    FileList = dir(fullfile(ImgLoc1, ['*.' Suffix]));

    for Info = FileList'
        FileName = Info.name;
        Img1 = imread(fullfile(ImgLoc1, FileName)) > 0;
        Img2 = imread(fullfile(ImgLoc2, FileName)) > 0;

        if any(Img1(:) ~= Img2(:))
            IsEqual = false;
            fprintf([FileName ' fail: ' num2str(sum(Img1(:))) ' ' num2str(sum(Img2(:))) ' ' num2str(sum(Img1(:))-sum(Img2(:))) '\n']);
            %break;
        end
    end
end