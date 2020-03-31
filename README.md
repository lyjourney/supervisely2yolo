Getting Started
---------------
```
$ git clone https://github.com/lyjourney/video2frame.git
$ cd video2frame
$ python3 conv2yolo.py --source_dir <name> [--dest_dir <name>] [--checker]
```
Usage example
-------------
```
$ python3 conv2yolo.py \
  --source_dir project \
  --dest_dir convert \
  --checker
```

Directory
---------
```
supervisely2yolo/
┣━ README.md
┣━ class.names
┣━ conv2yolo.py
┗━ project/
    ┗━ folder1
        ┣━ ann
        ┗━ [img] # optional (+ --checker)
    
    ┗━ folder2
        ┣━ ann
        ┗━ [img]
          :
```
Python package list
-------------------
* cv2
* shutil
* numpy
