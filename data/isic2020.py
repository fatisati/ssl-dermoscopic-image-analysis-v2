from data.image_dataset import ImageDataset


class ISIC2020(ImageDataset):
    def __init__(self, image_folder, label_file_path, image_size):
        image_col, label_col = 'image_name', 'target'
        super(ISIC2020, self).__init__(image_folder, label_file_path,
                 image_col, label_col, image_size)

    def __str__(self):
        return 'isic2020'
