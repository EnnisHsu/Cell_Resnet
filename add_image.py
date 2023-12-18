import os

if __name__=="__main__":
    train_file = open('train_03.txt','w')
    data_path = '/home/cell/datasets/cell/images/07分叶核/newfenyehe_trad/'
    dir_names = os.listdir(data_path)
    for dir in dir_names:
        new_dir = os.path.join('.',data_path[20:],dir)
        train_file.write(new_dir+' 07\n')
        