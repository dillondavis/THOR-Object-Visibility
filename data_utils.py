import pandas as pd
from pycocotools.coco import COCO

real_classes = ['Apple', 'Bowl', 'Bread', 'Butter Knife', 'Cabinet', 'Chair', 'Coffee Machine', 'Container', 'Egg', 'Fork', 'Fridge', 'Garbage Can', 'Knife', 'Lettuce', 'Microwave', 'Mug', 'Pan', 'Plate', 'Pot', 'Potato', 'Sink', 'Spoon', 'Stove Burner', 'Stove Knob', 'Table Top', 'Toaster', 'Tomato']
coco_classes = ['apple', 'bowl', None, 'knife', None, 'chair', None, None, None, 'fork', 'refrigerator', None, 'knife', None, 'microwave', 'cup', None, None, None, None, 'sink', 'spoon', None, None, 'dining table', 'toaster', None]


def get_similar_open_image_classes(real_classes):
    '''
    Writes OpenImage classes similar to real_classes to a file
    :param real_classes: List of class names to get similar OpenImage classes of
    '''

    datapath = '/Users/Dillon/UIUC/Research/OpenImage/class-descriptions.csv'
    classes = pd.read_csv(datapath, header=None, names=['id', 'class'])
    classes['class'] = classes['class'].map(lambda x: x.lower())

    f = open('classes.txt', 'w')
    for real_class in real_classes:
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

    print(len(classes))
    print(len(real_classes))
    print(len(coco_classes))
    f.close()

    return classes


def get_open_image_class_dict():
    '''
    :return: Dictionary mapping OpenImage class names to OpenImage class IDs
    '''

    datapath = '/Users/Dillon/UIUC/Research/OpenImage/class-descriptions.csv'
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
    for real_class, open_classes, coco_class in zip(real_classes, open_image_classes, coco_classes):
        if coco_class:
            f.write('{},{},{},{}\n'.format(coco_class, 'nan', 'coco', real_class))
        for open_class in open_classes:
            if open_class:
                f.write('{},{},{},{}\n'.format(open_class, open_image_class_dict[open_class], 'open', real_class))
    f.close()


def get_open_data(open_class_data):
    '''
    :param open_class_data: pandas DataFrame with necessary OpenImage class mappings
    :return: pandas DataFrame containing all images for necessary OpenImage classes
    '''

    datapath = '/Users/Dillon/UIUC/Research/OpenImage/data/train/annotations-human.csv'
    image_data = pd.read_csv(datapath)
    open_data = pd.merge(open_class_data, image_data, left_on='ClassID', right_on='LabelName')
    open_data.drop(['LabelName'], inplace=True, axis=1)

    return open_data


def get_coco_data():
    '''
    :return: pandas DataFrame containing all images for necessary COCO classes
    '''

    ann_file = '/Users/Dillon/UIUC/Research/coco/annotations/instances_train2017.json'
    coco = COCO(ann_file)
    columns = ['AltClass', 'ClassID', 'Source_x', 'RealClass', 'ImageID', 'Source_y', 'Confidence']
    data = []
    for coco_class, real_class in zip(coco_classes, real_classes):
        if coco_class:
            catIds = coco.getCatIds(catNms=[coco_class])
            imgIds = coco.getImgIds(catIds=catIds)
            for image_id in imgIds:
                data += [[coco_class, 'nan', 'coco', real_class, image_id, 'nan', 'nan']]

    return pd.DataFrame(data, columns=columns)


def build_class_map_dataset():
    open_image_classes = get_cleaned_open_image_classes()
    open_image_class_dict = get_open_image_class_dict()
    create_class_map_data(open_image_classes, open_image_class_dict)


def build_image_dataset(output_class_counts=False):
    classes = pd.read_csv('class_map.csv')
    open_class_data = classes[classes['Source'] == 'open']
    coco_class_data = classes[classes['Source'] != 'open']
    open_data = get_open_data(open_class_data)
    coco_data = get_coco_data()
    image_data = pd.concat([open_data, coco_data])
    image_data.to_csv('image_data.csv')

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
    build_class_map_dataset()
    build_image_dataset(True)
