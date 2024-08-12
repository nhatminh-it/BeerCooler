from torchvision import datasets
def load_class_names(train_folder):
    dataset = datasets.ImageFolder(train_folder)
    return dataset.classes

# Đường dẫn tới thư mục huấn luyện
train_folder = 'data/detected/train'

# Kiểm tra xem các lớp có được load đúng không
class_names = load_class_names(train_folder)
print("Loaded class names:", class_names)