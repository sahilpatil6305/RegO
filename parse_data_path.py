
# import os
# import os.path as osp
# from sklearn.model_selection import train_test_split
# # data_dir ='/scratch/zhiqiu/yfcc_dynamic_10/dynamic_300/images'
# # data_dir = "/Volumes/Aditya One/BTPs project/dataset/release_in_the_wild/dataset"




# # def list_all_files(rootdir):
# #     _files = []
# #     list_file = os.listdir(rootdir)
# #     for i in range(0,len(list_file)):
# #         path = os.path.join(rootdir,list_file[i])

# #         if os.path.isdir(path):
# #             _files.extend(list_all_files(path))
# #         if os.path.isfile(path):
# #              _files.append(path)
# #     return _files

# # def list_all_files(args,rootdir):

# #     train_list,test_list,all_list = [],[],[]
# #     bucket_list = os.listdir(rootdir)
# #     # bucket_list=list(filter(lambda a: 'bucket_' in a,bucket_list))
# #     if('0' in bucket_list):
# #         bucket_list.remove('0') # skip bucket 0, since it's for pretrain feature
# #     classes_list=  os.listdir(osp.join(rootdir,bucket_list[0]))
# #     if('clear25d' in args.split and 'BACKGROUND' in classes_list):
# #         classes_list.remove('BACKGROUND') # skip bucket 0, since it's for pretrain feature
# #     for bucket in bucket_list:
# #         for classes in classes_list:
      
# #             image_list=os.listdir(osp.join(rootdir,bucket,classes))
            
# #             image_list=list(map(lambda a: osp.join(osp.join(rootdir,bucket,classes,a)), image_list))
# #             image_list=image_list[:args.num_instance_each_class] # if background class have more image, we use only part of it
 
# #             try:
# #                 assert len(image_list)==args.num_instance_each_class
# #             except:
# #                 import pdb;pdb.set_trace()
# #                 print('a')
# #             train_subset,test_subset=train_test_split(image_list,test_size=args.test_split, random_state=args.random_seed)
# #             train_list.extend(train_subset)
# #             test_list.extend(test_subset)
# #             all_list.extend(image_list)
# #     import random
# #     random.seed(args.random_seed)
# #     random.shuffle(train_list)
# #     random.shuffle(test_list)
# #     random.shuffle(all_list)
# #     return train_list,test_list,all_list

# def list_all_files(args, rootdir):
#     import os, os.path as osp
#     from sklearn.model_selection import train_test_split
#     import random

#     train_list, test_list, all_list = [], [], []
#     classes_list = os.listdir(rootdir)  # ["fake", "real"]

#     for classes in classes_list:
#         class_path = osp.join(rootdir, classes)
#         if not osp.isdir(class_path):
#             continue

#         # pick only wav files
#         file_list = [osp.join(class_path, f) for f in os.listdir(class_path) if f.endswith(".wav")]

#         # limit
#         file_list = file_list[:args.num_instance_each_class]

#         if len(file_list) == 0:
#             print(f"⚠️ No files found in {class_path}")
#             continue

#         train_subset, test_subset = train_test_split(
#             file_list, test_size=args.test_split, random_state=getattr(args, "random_seed", 42)
#         )
#         train_list.extend(train_subset)
#         test_list.extend(test_subset)
#         all_list.extend(file_list)

#     random.seed(getattr(args, "random_seed", 42))
#     random.shuffle(train_list)
#     random.shuffle(test_list)
#     random.shuffle(all_list)

#     return train_list, test_list, all_list




# # def parse_data_path(args):

# #     class_list=args.class_list.split()
# #     # if available, use pre-split train/test, else, auto split the data_folder_path
# #     if(args.data_test_path !='' and args.data_train_path!=''):
# #         _, _,train_list= list_all_files(args,args.data_train_path)

# #         train_datasize=args.num_instance_each_class
# #         args.num_instance_each_class=args.num_instance_each_class_test

# #         _, _,test_list= list_all_files(args,args.data_test_path)
# #         all_list=train_list+test_list
# #         args.num_instance_each_class=train_datasize
# #     else:
# #         data_dir = args.data_folder_path
# #         print('parse data from {}'.format(data_dir))
# #         train_list, test_list,all_list= list_all_files(args,data_dir)
# #     os.makedirs("../{}/data_cache/".format(args.split),exist_ok=True)
# #     for stage in ['train','test','all']:
# #         if(stage=='train'):
# #             image_list=train_list
# #         elif(stage=='test'):
# #             image_list=test_list
# #         else:
# #             image_list=all_list
# #         # folder need to be like */folder/timestamp/class/image.png
# #         with open('../{}/data_cache/data_{}_path.txt'.format(args.split,stage) , 'w') as file:
# #             file.write("file class_index timestamp")
# #             for item in image_list:
# #                 name_list=item.split('/')
# #                 classes=name_list[-2]
# #                 if classes not in class_list:
# #                     continue
# #                 class_index=class_list.index(classes)
# #                 timestamp=name_list[-3]
# #                 # timestamp=name_list[-3].split('_')[-1] # since name is bucket_x
# #                 file.write("\n")
# #                 file.write(item+ " "+str(class_index)+" "+str(timestamp))
# #         print('{} parse path finish!'.format(stage))

# def parse_data_path(args):
#     import os, os.path as osp

#     class_list = args.class_list.split()
#     # if available, use pre-split train/test, else, auto split the data_folder_path
#     if (args.data_test_path != '' and args.data_train_path != ''):
#         _, _, train_list = list_all_files(args, args.data_train_path)

#         train_datasize = args.num_instance_each_class
#         args.num_instance_each_class = args.num_instance_each_class_test

#         _, _, test_list = list_all_files(args, args.data_test_path)
#         all_list = train_list + test_list
#         args.num_instance_each_class = train_datasize
#     else:
#         data_dir = args.data_folder_path
#         print('parse data from {}'.format(data_dir))
#         train_list, test_list, all_list = list_all_files(args, data_dir)

#     os.makedirs("../{}/data_cache/".format(args.split), exist_ok=True)

#     for stage in ['train', 'test', 'all']:
#         if stage == 'train':
#             image_list = train_list
#         elif stage == 'test':
#             image_list = test_list
#         else:
#             image_list = all_list

#         # folder need to be like */folder/timestamp/class/image.png
#         with open('../{}/data_cache/data_{}_path.txt'.format(args.split,stage), 'w') as file:
            
#             file.write("file\tclass_index\ttimestamp")  # tab-separated header
#             for item in image_list:
#                 name_list=item.split('/')
#                 classes=name_list[-2]
#                 if classes not in class_list:
#                     continue
#                 class_index=class_list.index(classes)
#                 timestamp=name_list[-3]
#                 file.write("\n")
#                 file.write(item + "\t" + str(class_index) + "\t" + str(timestamp))

#         print('{} parse path finish!'.format(stage))

import os
import os.path as osp
from sklearn.model_selection import train_test_split
import random

def list_all_files(args, rootdir):
    """
    List all audio files in the dataset, split into train/test/all.
    Only picks .wav files and limits per class to args.num_instance_each_class.
    """
    train_list, test_list, all_list = [], [], []

    # Sanitize and normalize the input path to avoid stray newlines or mixed separators
    if isinstance(rootdir, str):
        rootdir = rootdir.replace('\r', '').replace('\n', '').strip()
    rootdir = os.path.normpath(rootdir)

    try:
        entries = os.listdir(rootdir)
    except Exception as e:
        print(f"[list_all_files] Error listing directory for rootdir: {repr(rootdir)}")
        raise

    classes_list = [d for d in entries if osp.isdir(osp.join(rootdir, d))]
    if len(classes_list) == 0:
        print(f"⚠️ No classes found in {rootdir}")
        return train_list, test_list, all_list

    for cls in classes_list:
        class_path = osp.join(rootdir, cls)

        # pick only wav files
        file_list = [osp.join(class_path, f) for f in os.listdir(class_path) if f.endswith(".wav")]

        # limit per class ONLY if use_all_datasets is False
        use_all = getattr(args, "use_all_datasets", False)
        if not use_all:
             file_list = file_list[:args.num_instance_each_class]
        else:
             # If using all datasets, we might want to log this
             pass

        if len(file_list) == 0:
            print(f"⚠️ No audio files found in {class_path}")
            continue

        train_subset, test_subset = train_test_split(
            file_list, test_size=args.test_split, random_state=getattr(args, "random_seed", 42)
        )
        train_list.extend(train_subset)
        test_list.extend(test_subset)
        all_list.extend(file_list)

    random.seed(getattr(args, "random_seed", 42))
    random.shuffle(train_list)
    random.shuffle(test_list)
    random.shuffle(all_list)

    return train_list, test_list, all_list


# def parse_data_path(args):
#     """
#     Parse the dataset folder and generate data_train_path.txt,
#     data_test_path.txt, and data_all_path.txt for CLEARDataset.
#     """

#     class_list = args.class_list.split()

#     if args.data_train_path and args.data_test_path:
#         _, _, train_list = list_all_files(args, args.data_train_path)
#         train_datasize = args.num_instance_each_class
#         args.num_instance_each_class = args.num_instance_each_class_test
#         _, _, test_list = list_all_files(args, args.data_test_path)
#         all_list = train_list + test_list
#         args.num_instance_each_class = train_datasize
#     else:
#         print(f"Parsing data from {args.data_folder_path}")
#         train_list, test_list, all_list = list_all_files(args, args.data_folder_path)

#     cache_dir = osp.join("..", args.split, "data_cache")
#     os.makedirs(cache_dir, exist_ok=True)

#     for stage, file_list in zip(['train', 'test', 'all'], [train_list, test_list, all_list]):
#         txt_path = osp.join(cache_dir, f"data_{stage}_path.txt")
#         with open(txt_path, "w") as f:
#             f.write("file\tclass_index\tlabel\n")  # tab-separated header
#             for item in file_list:
#                 cls_name = osp.basename(osp.dirname(item))
#                 if cls_name not in class_list:
#                     continue
#                 class_index = class_list.index(cls_name)
#                 # For audio, timestamp can just be "0" or the class name if you want
#                 timestamp = "0"
#                 f.write(f"{item}\t{class_index}\t{timestamp}\n")

#         print(f"{stage} parse path finished! -> {txt_path}")

def parse_data_path(args):
    """
    Parse the dataset folder and generate data_train_path.txt,
    data_test_path.txt, and data_all_path.txt for CLEARDataset.
    """

    import os
    import os.path as osp

    class_list = args.class_list.split()

    if args.data_train_path and args.data_test_path:
        _, _, train_list = list_all_files(args, args.data_train_path)
        train_datasize = args.num_instance_each_class
        args.num_instance_each_class = args.num_instance_each_class_test
        _, _, test_list = list_all_files(args, args.data_test_path)
        all_list = train_list + test_list
        args.num_instance_each_class = train_datasize
    else:
        data_dir = args.data_folder_path
        print(f"Parsing data from {data_dir}")
        print(f"[debug] repr(data_dir) = {repr(data_dir)}")


        # Support two common layouts:
        # 1) dataset_root/<class>/*.wav
        # 2) dataset_root/{train,test,val}/<class>/*.wav
        train_sub = os.path.join(data_dir, 'train')
        test_sub = os.path.join(data_dir, 'test')
        val_sub = os.path.join(data_dir, 'val')

        print(f"[debug] train_sub={repr(train_sub)} exists={os.path.isdir(train_sub)}")
        print(f"[debug] test_sub={repr(test_sub)} exists={os.path.isdir(test_sub)}")
        if os.path.isdir(train_sub) and os.path.isdir(test_sub):
            # Collect lists from train/ and test/ separately
            t_train, _, _ = list_all_files(args, train_sub)
            t_test, _, _ = list_all_files(args, test_sub)
            train_list = t_train
            test_list = t_test
            all_list = train_list + test_list
            # Include val split if present (optional)
            if os.path.isdir(val_sub):
                v_train, v_test, v_all = list_all_files(args, val_sub)
                # treat val content as additional test data by default
                test_list.extend(v_all)
                all_list.extend(v_all)
        else:
            train_list, test_list, all_list = list_all_files(args, data_dir)

    cache_dir = osp.join("..", args.split, "data_cache")
    os.makedirs(cache_dir, exist_ok=True)

    for stage, file_list in zip(['train', 'test', 'all'], [train_list, test_list, all_list]):
        txt_path = osp.join(cache_dir, f"data_{stage}_path.txt")
        with open(txt_path, "w") as f:
            # Tab-separated header
            f.write("file\tclass_index\tlabel\n")
            for item in file_list:
                # Extract class from folder name
                cls_name = osp.basename(osp.dirname(item))
                if cls_name not in class_list:
                    continue
                class_index = class_list.index(cls_name)
                # Set label (timestamp or 0 for audio)
                label = "0"
                # Write using tab, ensures paths with spaces are safe
                f.write(f"{item}\t{class_index}\t{label}\n")

        print(f"{stage} parse path finished! -> {txt_path}")


def move_data_trinity(data_path, flag):
    """
    Stub function for data path handling.
    Returns the path unchanged as data is already in correct location.
    """
    return data_path
