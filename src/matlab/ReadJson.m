function val = ReadJson(FileLoc)
    fp  = fopen(FileLoc);
    raw = fread(fp, inf);
    str = char(raw');
    fclose(fp);
    val = jsondecode(str);
end