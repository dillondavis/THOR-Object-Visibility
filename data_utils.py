import os
import torch
import pandas as pd
import numpy as np
import requests
from scipy import misc
from PIL import Image
#from pycocotools.coco import COCO
from shutil import copyfile
from io import BytesIO

CLUSTER_ENV = True
COCO_ID_LENGTH = 12
REAL_CLASSES = ['Apple', 'Bowl', 'Bread', 'Butter Knife', 'Cabinet', 'Chair', 'Coffee Machine', 'Container', 'Egg', 'Fork', 'Fridge', 'Garbage Can', 'Knife', 'Lettuce', 'Microwave', 'Mug', 'Pan', 'Plate', 'Pot', 'Potato', 'Sink', 'Spoon', 'Stove Burner', 'Stove Knob', 'Table Top', 'Toaster', 'Tomato']
COCO_CLASSES = ['apple', 'bowl', None, 'knife', None, 'chair', None, None, None, 'fork', 'refrigerator', None, 'knife', None, 'microwave', 'cup', None, None, None, None, 'sink', 'spoon', None, None, 'dining table', 'toaster', None]
DATA_DIR = ('/data/ddavis14' if CLUSTER_ENV else '/Users/Dillon/UIUC/Research') + '/AllenAI-Object-Visibility'
IMAGE_DIR = DATA_DIR + '/images/training_data/obj_vis/train'
PROJECT_DIR = ('/home/nfs/ddavis14' if CLUSTER_ENV else '/Users/Dillon/UIUC/Research') + '/AllenAI-Object-Visibility'
DATA_UTIL_DIR = PROJECT_DIR + '/THOR-Object-Visibility'
OFFICIAL_CLASS_LIST = [
            'None', 'Apple', 'Bowl', 'Bread', 'ButterKnife', 'Cabinet', 'Chair',
            'CoffeeMachine', 'Container', 'Egg', 'Fork', 'Fridge', 'GarbageCan',
            'Knife', 'Lettuce', 'Microwave', 'Mug', 'Pan', 'Plate', 'Pot',
            'Potato', 'Sink', 'Spoon', 'StoveBurner', 'StoveKnob', 'TableTop',
            'Toaster', 'Tomato'
]
CLASS_ID_MAP = {name:i for i, name in enumerate(OFFICIAL_CLASS_LIST)}


def get_similar_open_image_classes(REAL_CLASSES):
    '''
    Writes OpenImage classes similar to REAL_CLASSES to a file
    :param REAL_CLASSES: List of class names to get similar OpenImage classes of
    '''

    datapath = DATA_DIR + '/OpenImage/class-descriptions.csv'
    classes = pd.read_csv(datapath, header=None, names=['id', 'class'])
    classes['class'] = classes['class'].map(lambda x: x.lower())

    f = open('classes.txt', 'w')
    for real_class in REAL_CLASSES:
        f.write('Real Class: {}\n'.format(real_class))
        for class_tok in real_class.split():
            similar_classes = classes[classes['class'].str.contains(class_tok.lower())]
            for sim_class in similar_classes['class']:
                f.write("\'" + sim_class + "\', ")
        f.write('\n\n')
    f.close()


def get_cleaned_open_image_classes():
    '''
    Create a list of lists containing OpenImage classes for each proprietary class
    from cleaned class file
    '''

    f = open('cleaned_open_image_classes.txt', 'r')
    classes = []
    for line in f:
        if line and 'Real Class' not in line:
            if 'None' in line:
                classes += [[]]
            else:
                names = line.strip()
                classes += [[name.strip() for name in names.split(',') if name.strip()]] if names else []

    f.close()

    return classes


def get_open_image_class_dict():
    '''
    :return: Dictionary mapping OpenImage class names to OpenImage class IDs
    '''

    datapath = DATA_DIR + '/OpenImage/class-descriptions.csv'
    classes = pd.read_csv(datapath, header=None, names=['id', 'class'])
    classes['class'] = classes['class'].map(lambda x: x.lower())

    return classes.set_index('class').to_dict()['id']


def create_class_map_data(open_image_classes, open_image_class_dict):
    '''
    Create dataset mapping COCO and OpenImage classes to our proprietary classes
    :param open_image_classes: list of lists containing OpenImage classes for each proprietary class
    :param open_image_class_dict: dictionary mapping OpenImage class name to OpenImage class ID
    '''

    f = open('class_map.csv', 'w')
    f.write('AltClass,ClassID,Source,RealClass\n')
    for real_class, open_classes, coco_class in zip(REAL_CLASSES, open_image_classes, COCO_CLASSES):
        if coco_class:
            f.write('{},{},{},{}\n'.format(coco_class, 'nan', 'coco', real_class))
        for open_class in open_classes:
            if open_class:
                f.write('{},{},{},{}\n'.format(open_class, open_image_class_dict[open_class], 'open', real_class))
    f.close()


def get_open_ids(open_class_data):
    '''
    :param open_class_data: pandas DataFrame with necessary OpenImage class mappings
    :return: pandas DataFrame containing all images for necessary OpenImage classes
    '''

    datapath = DATA_DIR + '/OpenImage/data/train/annotations-human.csv'
    image_data = pd.read_csv(datapath)
    open_data = pd.merge(open_class_data, image_data, left_on='ClassID', right_on='LabelName')
    open_data.drop(['LabelName'], inplace=True, axis=1)
    open_data = open_data[open_data['Confidence'] != 0]

    return open_data


def get_coco_ids():
    '''
    :return: pandas DataFrame containing all images for necessary COCO classes
    '''

    ann_file = DATA_DIR + '/coco/annotations/instances_train2017.json'
    coco = COCO(ann_file)
    columns = ['AltClass', 'ClassID', 'Source_x', 'RealClass', 'ImageID', 'Source_y', 'Confidence']
    data = []
    for coco_class, real_class in zip(COCO_CLASSES, REAL_CLASSES):
        if coco_class:
            catIds = coco.getCatIds(catNms=[coco_class])
            imgIds = coco.getImgIds(catIds=catIds)
            for image_id in imgIds:
                data += [[coco_class, 'nan', 'coco', real_class, image_id, 'nan', -1]]

    return pd.DataFrame(data, columns=columns)


def get_open_images(id_data, class_limit):
    id_data = id_data[['ImageID', 'RealClass']].groupby(id_data['RealClass']).head(class_limit)
    image_data = pd.read_csv(DATA_DIR + '/OpenImage/data/train/images.csv')
    image_data = image_data[['ImageID', 'OriginalURL', 'OriginalLandingURL']]
    image_data = pd.merge(id_data, image_data, left_on='ImageID', right_on='ImageID').groupby(image_data['ImageID'])
    output_image_dir = IMAGE_DIR
    output_image_file = output_image_dir + '/{}_open.pt'
    invalid = 0

    for image_id, group in image_data:
        classes = group['RealClass'].as_matrix()
        obj_vis = np.array([1 if name in classes else 0 for name in OFFICIAL_CLASS_LIST], dtype=np.uint8)
        image_url = list(group['OriginalURL'])[0]
        image_file = output_image_dir + '/{}.jpg'.format(image_id)
        image_bytes = BytesIO(requests.get(image_url).content)
        try:
            image = np.array(Image.open(image_bytes)).astype(np.uint8)
            if len(image.shape) == 3:
                torch.save({'frame':np.array(image).astype(np.uint8), 'obj_vis':obj_vis}, output_image_file.format(str(image_id)))
            else:
                invalid += 1
        except:
            print("invalid")
            invalid += 1
    print("TOTAL INVALID: {}".format(invalid))


def get_coco_images(id_data, class_limit):
    id_data = id_data[['ImageID', 'RealClass']].groupby(id_data['RealClass']).head(class_limit).groupby(id_data['ImageID'])
    coco_image_file = DATA_DIR + '/coco/images/{}.jpg'
    output_image_dir = IMAGE_DIR 
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    output_image_file = output_image_dir + '/{}_coco.pt'
    invalid = 0

    for image_id, group in id_data:
        classes = group['RealClass'].as_matrix()
        obj_vis = np.array([1 if name in classes else 0 for name in OFFICIAL_CLASS_LIST], dtype=np.uint8)
        id_str = pad_img_num(image_id, COCO_ID_LENGTH)
        image = misc.imread(coco_image_file.format(id_str)).astype(np.uint8)
        if len(image.shape) == 3:
            torch.save({'frame':image, 'obj_vis':obj_vis}, output_image_file.format(id_str))
        else:
            invalid += 1

    print("TOTAL INVALID: {}".format(invalid))


def pad_img_num(num, total_digits):
    num_str = ''
    for _ in range(total_digits - len(str(num))):
        num_str += '0'

    return num_str + str(num)


def build_image_dataset():
    id_data = pd.read_csv('id_data.csv')
    open_id_data = id_data[id_data['Source_x'] == 'open']
    coco_id_data = id_data[id_data['Source_x'] != 'open']
    get_coco_images(coco_id_data, 50)
    get_open_images(open_id_data, 50)


def build_class_map_dataset():
    open_image_classes = get_cleaned_open_image_classes()
    open_image_class_dict = get_open_image_class_dict()
    create_class_map_data(open_image_classes, open_image_class_dict)


def build_id_dataset(output_class_counts=False):
    classes = pd.read_csv('class_map.csv')
    open_class_data = classes[classes['Source'] == 'open']
    coco_class_data = classes[classes['Source'] != 'open']
    open_data = get_open_ids(open_class_data)
    coco_data = get_coco_ids()
    image_data = pd.concat([open_data, coco_data])
    image_data.to_csv('id_data.csv')

    if output_class_counts:
        groups = open_data['ImageID'].groupby(open_data['RealClass'])
        print('OpenImage Counts')
        print(groups.size())
        print('')
        groups = coco_data['ImageID'].groupby(coco_data['RealClass'])
        print('COCO Counts')
        print(groups.size())
        print('')
        groups = image_data['ImageID'].groupby(image_data['RealClass'])
        print('Combined Counts')
        print(groups.size())
        print('')


if __name__ == '__main__':
    #build_class_map_dataset()
    #build_id_dataset(True)
    build_image_dataset()
