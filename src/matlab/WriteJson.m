function WriteJson(FileLoc, Content)
    fp = fopen(FileLoc, 'w');
    Content = jsonencode(Content, 'PrettyPrint', true);
    fprintf(fp, Content);
    fclose(fp);
end