import torch
from Utils import *
from Model import Network
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from scipy.stats import spearmanr as SROCC


def test_model():
    ### Win5
    load_all_model_path = './PreTrainedModels/Win5/'
    valset_dir = './Datasets/Win5_MLI_7x32x32/'
    dataset_name = 'Win5'
    scene_list = ['Bikes', 'dishes', 'Flowers', 'greek', 'museum', 'Palais_du_Luxembourg', 'rosemary', 'Sphynx', 'Swans_1', 'Vespa']
    test_scene_num = 2
    distorted_num = 5
    scene_num = 10
    dist = 'NN'  # HEVC, JPEG2000, LN, NN
    Dist_DICT = []

    ### NBU
    # load_all_model_path = './PreTrainedModels/NBU/'
    # valset_dir = './Datasets/NBU_MLI_7x32x32/'
    # dataset_name = 'NBU'
    # scene_list = ['I01R0', 'I02R0', 'I03R0', 'I04R0', 'I05R0', 'I06R0', 'I07R0',
    #               'I08R0', 'I09R0', 'I10R0', 'I11R0', 'I12R0', 'I13R0', 'I14R0']
    # test_scene_num = 2
    # distorted_num = 3
    # scene_num = 14
    # Dist_DICT = {'NN': [1, 2, 3], 'BI': [4, 5, 6], 'EPICNN': [7, 8, 9], 'Zhang': [90, 91, 92], 'VDSR': [93, 94, 95]}
    # dist = 'VDSR'

    ### SHU
    # load_all_model_path = './PreTrainedModels/SHU/'
    # valset_dir = './Datasets/SHU_MLI_7x32x32/'
    # dataset_name = 'SHU'
    # scene_list = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8']
    # test_scene_num = 2
    # distorted_num = 6
    # scene_num = 8
    # Dist_DICT = {'GAUSS':1,'JPEG2000':2,'JPEG':3,'MotionBlur':4,'WhiteNoise':5}
    # dist = 'GAUSS'

    device = 'cuda:0'
    net = Network().to(device)
    cudnn.benchmark = True

    all_model = os.listdir(load_all_model_path)
    label_list = np.zeros([test_scene_num * distorted_num, len(all_model)])
    data_list = np.zeros([test_scene_num * distorted_num, len(all_model)])
    val_SRCC_all = []
    test_scene_id_list = []
    for a in range(scene_num):
        for b in range(a + 1, scene_num):
            test_scene_id_list.append([a, b])

    for id, model_name in enumerate(test_scene_id_list):
        load_model_path = load_all_model_path + '/' + \
                          str(model_name[0]) + '_' + str(model_name[1]) + '/MAFBLiF_epoch30.pth.tar'
        model = torch.load(load_model_path, map_location={'cuda:0': device})
        net.load_state_dict(model['state_dict'])
        net.eval()
        index = 0
        test_scene_id = [int(model_name[0]), int(model_name[1])]
        for test_scene in test_scene_id:
            image_path = valset_dir + '/' + scene_list[test_scene]
            image_list = os.listdir(image_path)
            for test_image in image_list:
                if dataset_name == 'Win5':
                    if 'EPICNN' in test_image: continue
                    if dist == 'JPEG2000':
                        if 'JPEG' not in test_image: continue
                    else:
                        if dist not in test_image: continue
                elif dataset_name == 'NBU':
                    if int(test_image.split('R')[1]) not in Dist_DICT[dist]: continue
                elif dataset_name == 'SHU':
                    if int(test_image[1]) != Dist_DICT[dist]: continue
                else:
                    print("=> no found dataset name")
                patch_path = valset_dir + '/' + scene_list[test_scene] + '/' + test_image
                patch_list = os.listdir(patch_path)
                output_list = []
                VS_list = []
                for val_patch in patch_list:
                    if int(val_patch[:-3]) < int(len(patch_list) * 0.4):
                        continue
                    each_patch_path = patch_path + '/' + val_patch
                    with h5py.File(each_patch_path, 'r') as hf:
                        label = np.array(hf.get('score_label'))
                        data = np.array(hf.get('MLI'))
                        data = data / 255
                        data = np.expand_dims(data, axis=0)
                        data = np.expand_dims(data, axis=0)
                        data = torch.from_numpy(data.copy())
                        data = Variable(data).to(device)
                        VS = np.array(hf.get('VS'))
                    with torch.no_grad():
                        out_score = net(data)
                    output_list.append(out_score.cpu().numpy().item() * VS.item())
                    VS_list.append(VS.item())
                label_list[index, id] = label.item()
                data_list[index, id] = sum(output_list) / sum(VS_list)
                index += 1

        val_SRCC = SROCC(data_list[:, id], label_list[:, id]).correlation
        val_SRCC_all.append(val_SRCC)
        print(test_scene_id)
        print('SROCC :----    %f' % val_SRCC)
    print('Average SROCC :----   %f' % np.mean(val_SRCC_all))

    # save in h5 file and test in matlab
    f = h5py.File('./Results/MAFBLiF_result_' + dataset_name + '_' + dist + '.h5', 'w')
    f.create_dataset('predict_data', data=data_list)
    f.create_dataset('score_label', data=label_list)
    f.close()


if __name__ == '__main__':
    test_model()
