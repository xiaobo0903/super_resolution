import torch
import os
import utility
import data, time
import model
import loss
import shutil
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():

    global model
    from sr_video import sr_video
    model = model.Model(args, checkpoint)

    vspath = args.dir_sour
    vtpath = args.dir_tmp

    if not os.path.exists(vspath):
        os.makedirs(vspath)             

    if not os.path.exists(vtpath):
        os.makedirs(vtpath)             

    while True:
    
        lists = os.listdir(vspath)  #列出目录的下所有文件和文件夹保存到lists
        lists.sort(key=lambda fn: os.path.getmtime(vspath + "/" + fn), reverse=True) # 按时间排序
        
        for alist in lists:
            file_new = os.path.join(vspath, alist)      # 获取最新的文件保存到file_new

            if os.path.isfile(file_new): 
                shutil.move(file_new, vtpath)  # 把源视频文件移动到临时目录中
                file_new1 = os.path.join(vtpath, alist)      # 获取最新的文件保存到file_new
                args.dir_demo = file_new1
                sr = sr_video(args, model, checkpoint)
                sr.vscale()

            continue

        time.sleep(1)
    #sr.vscale()

if __name__ == '__main__':
    main()
