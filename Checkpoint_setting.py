# Modify this folder. Choose how many model you want to ensemble
# Format: list of
#   {"ckptname" : time stamp string, "topk" : top k pth files in that ckpt} 
ckpt_list = [
    {"ckptname" : 'mix_1/2022-05-18-01-05-57', "topk" : 1},
    {"ckptname" : 'mix_1/2022-05-18-12-16-15', "topk" : 5},
    {"ckptname" : 'mix_1/2022-05-18-17-00-12', "topk" : 1},
    {"ckptname" : 'mix_1/2022-05-19-00-16-59', "topk" : 7},
    {"ckptname" : 'mix_1/2022-05-25-15-38-19', "topk" : 1},
    {"ckptname" : '0531_4fold/2022-05-31-15-40-08', "topk" : 1},
    {"ckptname" : '0531_4fold/2022-05-31-15-42-28', "topk" : 1},
    {"ckptname" : '0601_4fold/2022-05-31-15-28-40', "topk" : 1},
    {"ckptname" : '0601_4fold/2022-05-31-15-31-29', "topk" : 1}
]
