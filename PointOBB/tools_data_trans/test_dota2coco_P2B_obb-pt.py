import dota_utils as util
import os
import cv2
import json
from PIL import Image
import shutil

wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']

wordname_16 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']

word_name_dior = ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
         'chimney', 'expressway-service-area', 'expressway-toll-station',
         'dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
         'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle',
         'windmill']

wordname_star = ['ship','boat','crane','goods_yard','tank','storehouse','breakwater','dock',
        'airplane','boarding_bridge','runway','taxiway','terminal','apron','gas_station',
        'truck','car','truck_parking','car_parking','bridge','cooling_tower','chimney',
        'vapor','smoke','genset','coal_yard','lattice_tower', 'substation', 'wind_mill',
        'cement_concrete_pavement', 'toll_gate', 'flood_dam', 'gravity_dam', 'ship_lock',
        'ground_track_field','basketball_court','engineering_vehicle', 'foundation_pit', 
        'intersection', 'soccer_ball_field','tennis_court','tower_crane','unfinished_building',
        'arch_dam','roundabout','baseball_diamond','stadium','containment_vessel']

wordname_fair = ['Boeing737', 'Boeing747', 'Boeing777', 'Boeing787', 'C919', 'A220',
                'A321', 'A330', 'A350', 'ARJ21', 'Passenger Ship', 'Motorboat',
                'Fishing Boat', 'Tugboat', 'Engineering Ship', 'Liquid Cargo Ship',
                'Dry Cargo Ship', 'Warship', 'Small Car', 'Bus', 'Cargo Truck',
                'Dump Truck', 'Van', 'Trailer', 'Tractor', 'Excavator',
                'Truck Tractor', 'Basketball Court', 'Tennis Court', 'Football Field',
                'Baseball Field', 'Intersection', 'Roundabout', 'Bridge']

wordname_rsar = ['ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor']

def DOTA2COCOTrain(srcpath, destfile, cls_names, difficult='2'):

    # DIOR
    # imageparent = os.path.join(srcpath, 'JPEGImages-trainval')  
    # labelparent = os.path.join(srcpath, 'labelTxt_obb_pt_trainval')
    # # DOTA
    imageparent = os.path.join(srcpath, 'images')
    labelparent = os.path.join(srcpath, 'labelTxt_obb_pt_train_viaobb_v1.0')

    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1
    with open(destfile, 'w') as f_out:
        filenames = util.GetFileFromThisRootDir(labelparent)
        for file in filenames:
            basename = util.custombasename(file)
            basetxtname = basename + '.txt'
            txt_path = os.path.join(labelparent, basetxtname)
            # image_id = int(basename[1:])

            imagepath = os.path.join(imageparent, basename + '.png') # DOTA
            # imagepath = os.path.join(imageparent, basename + '.jpg')  # DIOR

            # if not os.path.exists(imagepath):  # move testset in DIOR
            #     shutil.move(txt_path, os.path.join('DIOR/labelTxt_obb_pt_test/', basetxtname))
            #     continue

            img = cv2.imread(imagepath)
            height, width, c = img.shape

            single_image = {}
            single_image['file_name'] = basename + '.png'  # DOTA
            # single_image['file_name'] = basename + '.jpg'  # DIOR
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            # annotations
            txtpath = os.path.join(labelparent, basetxtname)
            with open(txtpath, 'r') as f_in:
                lines = f_in.readlines()
                splitlines = [x.strip().split(' ') for x in lines]
                boxes = []
                for i, splitline in enumerate(splitlines):
                    
                    x1 = float(splitline[0])
                    y1 = float(splitline[1])
                    x2 = float(splitline[2])
                    y2 = float(splitline[3])
                    x3 = float(splitline[4])
                    y3 = float(splitline[5])
                    x4 = float(splitline[6])
                    y4 = float(splitline[7])
                    point_x = float(splitline[8])
                    point_y = float(splitline[9])

                    class_name = splitline[10]
                    difficulty = splitline[11]
                    # class_name = " ".join(splitline[10:-1]) # for fair
                    # difficulty = splitline[-1]  # for fair
                    # class_name = class_name.lower() # DIOR
                    assert class_name in cls_names

                    single_obj = {}
                    single_obj['category_id'] = cls_names.index(class_name) + 1
                    single_obj['segmentation'] = []
                    single_obj['segmentation'].append([x1,y1,x2,y2,x3,y3,x4,y4])
                    single_obj['iscrowd'] = 0

                    x_min = min(x1,x2,x3,x4)
                    y_min = min(y1,y2,y3,y4)
                    x_max = max(x1,x2,x3,x4)
                    y_max = max(y1,y2,y3,y4)
                    width, height = x_max - x_min, y_max - y_min
                    area = width * height
                    single_obj['area'] = area

                    # pseudo hbox (as point label)
                    width_ =16.0
                    height_ = 16.0
                    # CPR/P2B 
                    single_obj['point'] = point_x, point_y
                    single_obj['true_rbox'] = x1, y1, x2, y2, x3, y3, x4, y4
                    single_obj['bbox'] = point_x-8.0, point_y-8.0, width_, height_
                    single_obj['image_id'] = image_id
                    single_obj['id'] = inst_count
                    data_dict['annotations'].append(single_obj)
                
                    inst_count = inst_count + 1

            image_id = image_id + 1
        
            print(f'finish{file}')
        json.dump(data_dict, f_out)
        print('done!')

def DOTA2COCOTest(srcpath, destfile, cls_names):
    imageparent = os.path.join(srcpath, 'images')
    data_dict = {}

    data_dict['images'] = []
    data_dict['categories'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    image_id = 1
    with open(destfile, 'w') as f_out:
        filenames = util.GetFileFromThisRootDir(imageparent)
        for file in filenames:
            basename = util.custombasename(file)
            imagepath = os.path.join(imageparent, basename + '.png')
            img = Image.open(imagepath)
            height = img.height
            width = img.width

            single_image = {}
            single_image['file_name'] = basename + '.png'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            image_id = image_id + 1
        json.dump(data_dict, f_out)

if __name__ == '__main__':
    DOTA2COCOTrain(r'/datasets/mmrotate_data/RSAR/test/',
                   r'/datasets/mmrotate_data/RSAR/test/test_rbox_pt_P2Bfmt.json',
                   wordname_rsar)

