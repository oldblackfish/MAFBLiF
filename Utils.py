import os
import h5py
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor


class MyTrainSetLoader_Kfold(Dataset):
    def __init__(self, dataset_dir, test_scene_id):
        super(MyTrainSetLoader_Kfold, self).__init__()
        self.dataset_dir = dataset_dir
        scene_list = ['Bikes', 'dishes', 'Flowers', 'greek', 'museum', 'Palais_du_Luxembourg', 'rosemary', 'Sphynx', 'Swans_1', 'Vespa']

        scene_list.pop(test_scene_id[0])
        scene_list.pop(test_scene_id[1] - 1)
        all_patch_path = []
        for scene in scene_list:
            distorted_scene_list = os.listdir(dataset_dir + '/' + scene)
            for distorted_scene in distorted_scene_list:
                distorted_path_list = os.listdir(dataset_dir + '/' + scene + '/' + distorted_scene)
                for distorted_path in distorted_path_list:
                    all_patch_path.append(scene + '/' + distorted_scene + '/' + distorted_path)
        self.all_patch_path = all_patch_path
        self.item_num = len(self.all_patch_path)

    def __getitem__(self, index):
        all_patch_path = self.all_patch_path
        dataset_dir = self.dataset_dir
        file_name = dataset_dir + '/' + all_patch_path[index]
        with h5py.File(file_name, 'r') as hf:
            data = np.array(hf.get('MLI'))
            data = data / 255
            score_label = np.array(hf.get('score_label'))
        return ToTensor()(data.copy()), ToTensor()(score_label.copy())

    def __len__(self):
        return self.item_num
