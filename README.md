Getting Started
---------------
```
1.  Make annotion on the https://supervise.ly/

2.  $ git clone https://github.com/lyjourney/video2frame.git
    $ cd supervisely2yolo
    $ python3 conv2yolo.py --source_dir <name> [--dest_dir <name>] [--checker]
```
Usage Example
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
    ┣━ folder1
    ┃   ┣━ ann
    ┃   ┗━ [img] # optional (+ --checker)
    ┃
    ┗━ folder2
        ┣━ ann
        ┗━ [img]
          :
```
Python Package List
-------------------
* cv2
* shutil
* numpy
