from types import SimpleNamespace
from load_dataset import CLEARDataset

# Define args with required attributes
args = SimpleNamespace(
    num_instance_each_class=100,   # example, apne dataset ke hisaab se adjust karo
    class_list="fake real",        # apne dataset ke classes
    split="release_in_the_wild",   # folder name for cache
    test_split=0.2,                # train/test split ratio
    random_seed=42,
    data_folder_path="/Volumes/AdityaOne/BTPsproject/dataset/release_in_the_wild/dataset",
    data_train_path="",
    data_test_path="",
    timestamp=1  
)

# Define num_classes from class_list
args.num_classes = len(args.class_list.split())

# Load datasets
train_dataset = CLEARDataset(args, data_txt_path='/Volumes/AdityaOne/BTPsproject/dataset/release_in_the_wild/data_cache/data_train_path.txt', stage='train')
test_dataset  = CLEARDataset(args, data_txt_path='/Volumes/AdityaOne/BTPsproject/dataset/release_in_the_wild/data_cache/data_test_path.txt', stage='test')

print("Number of train samples:", len(train_dataset))
print("Number of test samples:", len(test_dataset))
