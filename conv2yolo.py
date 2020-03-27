import os
import sys
import cv2
import numpy as np
import argparse
import json
import shutil


class STOY:

    def __init__(self, source_dir, dest_dir, labeler):
        self.source = source_dir
        self.dest = dest_dir

        self.labeler = labeler

        self.source_list = []

        cls = open('class.names', "r")
        CLASSES = cls.readlines()
        CLASSES = [_.split('\n')[0] for _ in CLASSES]
        self.class_to_ind = dict(zip(CLASSES, range(len(CLASSES))))
        #print(CLASSES)


    def find_source_list(self):
        source_dirs = os.listdir(self.source)
        source_list = np.array([], dtype = str)

        for source_dir in source_dirs:
            if len(source_dir.split('.')) == 1:
                source_list = np.append(source_list, str(source_dir))

        for dir in source_list:
            dir = os.path.join(self.source, dir, "ann")
            if not os.path.exists(dir):
                source_list = np.delete(source_list,
                            np.where(source_list == dir.split('/')[-2]))

                print(f"\nCan not find '{dir}' directory.\n")
                continue

        self.source_list = \
            [os.path.join(self.source, _, "ann") for _ in source_list]


    def conv2yolo(self, json, txt_name, label_path):
        lab_p = os.path.join(label_path, txt_name.split('.json')[0])
        txt_file = open(lab_p.replace(".jpg", ".txt"), 'w')

        w = json['size']['width']
        h = json['size']['height']

        if not json['objects']:
            print("Error: " + lab_p.replace(".jpg",".txt"), "is empty\n")
            sys.exit()

        for object in json['objects']:
            cls = self.class_to_ind[object['classTitle']]

            #(left, top), (right, bottom)
            (x1, y1), (x2, y2) = object['points']['exterior']
            x_cnt = ((x1 + x2) / 2) / w
            y_cnt = ((y1 + y2) / 2) / w
            width = abs((x2 - x1) / w)
            height = abs((y2 - y1) / h)

            width = 0 if width < 0 else width
            height = 0 if height < 0 else height
            width = w if width > w else width
            height = h if height > h else height

            txt_file.write(
                f"{cls} {x_cnt:.5f} {y_cnt:.5f} {width:.5f} {height:.5f}\n"
            )

        txt_file.close()
        return lab_p



    def run(self):
        self.find_source_list()

        if os.path.exists(self.dest):
            shutil.rmtree(self.dest)
        os.makedirs(self.dest)

        print(f"\n'{self.dest}' is created.")

        all_path = open(os.path.join(self.dest, "train.txt"), 'w')

        for dir in self.source_list:
            dir_path = os.path.join(self.dest, dir.split('/')[-2])
            label_path = os.path.join(dir_path, 'labels')
            images_path = label_path.replace('labels', 'images')

            os.makedirs(dir_path)
            os.makedirs(label_path)
            os.makedirs(images_path)

            for json_name in os.listdir(dir):
                with open(f"{dir}/{json_name}","r") as source_json:
                    source_json = json.load(source_json)
                    lab_p = self.conv2yolo(source_json, json_name, label_path)

                all_path.write(lab_p.replace("labels", "images") + '\n')

        all_path.close()

        os.system(f"chmod 777 -R {self.dest}")

        print("Annotation convert was done successfully.\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--source_dir", default = None, type = str,
            help = "source(project) directory")

    parser.add_argument("--dest_dir", default = "convert", type = str,
            help = "destination directory")

    args = parser.parse_args()

    if args.source_dir is None:
        print("\nError: Folder is not exists.")
        print("Please check your source_dir.\n")
        sys.exit()

    stoy = STOY(args.source_dir, args.dest_dir)

    stoy.run()
