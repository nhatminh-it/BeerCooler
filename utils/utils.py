import os

def get_classes(classes_folder_path: str) -> list:
    '''
    Function to load classes name from folder include logos images follow name(sort a-z).
    :param classes_folder_path:
    :return: list of labels
    '''
    labels = [os.path.splitext(f.name)[0] for f in os.scandir(classes_folder_path) if f.is_file()]
    labels = [label for label in labels if label != '.DS_Store']
    return sorted(labels)