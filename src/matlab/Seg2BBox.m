function Seg2BBox(Src, Dest)
    FileList = dir(fullfile(Src, '*.png'));
    JsonTemp = '{';
    for k = 1 : numel(FileList)
        FileName = FileList(k).name;
        imgIn = imread(fullfile(Src, FileName));
        imgIn = imgIn(:,:,1) > 0.5;
        imgSize = size(imgIn);

        CC_Cell = bwconncomp(imgIn).PixelIdxList;
        BBoxs   = strings(numel(CC_Cell), 5);

        for c = 1 : numel(CC_Cell)
            [Row, Col] = ind2sub(imgSize, CC_Cell{c});
            ConfScore  = rand(1);
            ConfScore  = 0.75 * ConfScore + 1 * (1-ConfScore);
            BBoxs(c,:) = [min(Col) min(Row) max(Col) max(Row) round(ConfScore,5)];
        end
        BBoxs(:,end) = pad(BBoxs(:,end), 7, 'right', '1');
        BBoxs = double(BBoxs);
        if size(BBoxs, 1) == 1
            ThisData = ['"' FileName(1:end-4) '.jpg":[' jsonencode(BBoxs) ']'];
        else
            ThisData = ['"' FileName(1:end-4) '.jpg":' jsonencode(BBoxs)];
        end
        
        JsonTemp = [JsonTemp ThisData ','];
    end
    JsonTemp = [JsonTemp(1:end-1) '}'];

    fp = fopen([Dest '.json'], 'w');
    fwrite(fp, JsonTemp);
    fclose(fp);
end
