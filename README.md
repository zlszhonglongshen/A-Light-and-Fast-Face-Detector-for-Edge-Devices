## A Light and Fast Face Detector for Edge Devices

## Recent Update
* `2019.07.25` This repos is first online.

## Introduction
This repo releases the source code of paper xxxxx (link to citation). Our paper presents a light and fast face detector (**LFFD**) for edge devices.
LFFD considerably balances both accuracy and latency, resulting in small model size, fast inference speed while achieving excellent accuracy. In
practical, we have deployed it in cloud and edge devices (like NVIDIA Jetson series and ARM-based embedding system). The comprehensive performance
of LFFD is robust enough to support our applications.

In fact, our method is **_a general detection framework that applicable to one class detection_**, such as face detection, pedestrian detection, 
head detection, vehicle detection and so on. In general, a object class, whose average ratio of the longer side and the shorter side is 
less than 5, is appropriate to apply our framework for detection.

#### Performance
We train LFFD on train set of WIDER FACE benchmark. All methods are evaluated on val/test sets under the SIO schema (please
refer to the paper for details).

Results on val set of WIDER FACE. The values in () are results from the original papers:

Method|Easy Set|Medium Set|Hard Set
------|--------|----------|--------
DSFD  |0.949(0.966)|0.936(0.957)|0.850(0.904)   
PyramidBox|0.937(0.961)|0.927(0.950)|0.867(0.889)
S3FD  |0.923(0.937)|0.907(0.924)|0.822(0.852)
SSH   |0.921(0.931)|0.907(0.921)|0.702(0.845)
FaceBoxes|0.840    |0.766       |0.395
FaceBoxes3.2×|0.798|0.802       |0.715
**LFFD**|0.910     |0.880       |0.780

Results on test set of WIDER FACE. The values in () are results from the original papers:

Method|Easy Set|Medium Set|Hard Set
------|--------|----------|--------
DSFD  |0.947(0.960)|0.934(0.953)|0.845(0.900)   
PyramidBox|0.926(0.956)|0.920(0.946)|0.862(0.887)
S3FD  |0.917(0.928)|0.904(0.913)|0.821(0.840)
SSH   |0.919(0.927)|0.903(0.915)|0.705(0.844)
FaceBoxes|0.839    |0.763       |0.396
FaceBoxes3.2×|0.791|0.794       |0.715
**LFFD**|0.896     |0.865       |0.770


## Getting Started
We implement the proposed method using MXNet Module API.
#### Prerequirements
* Python>=3.5
* numpy>=1.15
* MXNet>=1.4 ([install guide](http://mxnet.incubator.apache.org/versions/master/install/index.html?platform=Linux&language=Python&processor=GPU))
* cv2=3.x (pip3 install opencv-python==3.4.5.20, other version should work as well)

> Tips: 
  * use MXNet with cudnn.
  * build numpy from source with OpenBLAS. This will improve the training efficiency.
  * make sure cv2 links to libjpeg-turbo, not libjpeg.

#### Advanced

#### Tips for lower latency 

## Citation

## To Do List
