# A Light and Fast Face Detector for Edge Devices

## Recent Update
* `2019.07.25` This repos is first online. Face detection related content is released.

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

In the paper, we use three hardware platforms for latency evaluation: NVIDIA GTX TITAN Xp, NVIDIA TX2 and 
Rasberry Pi 3 Model B+ (ARM A53). In addition, we also involve some other hardwares: NVIDIA GTX 1060 (laptop), 
NVIDIA RTX 2080TI, NVIDIA Jetson Nano, RK3288 (ARM A17).

We report the time latency of inference only (for NVIDIA hardwares, data transfer is included), excluding
pre-processing and post-processing. The batchsize is set to 1 for all evaluations.

Latency on NVIDIA GTX TITAN Xp (MXNet+CUDA 9.0+CUDNN7.1) presented in the paper:

Resolution->|640×480|1280×720|1920×1080|3840×2160
------------|-------|--------|---------|---------
DSFD|78.08ms(12.81 FPS)|187.78ms(5.33 FPS)|392.82ms(2.55 FPS)|1562.50ms(0.64 FPS)
PyramidBox|50.51ms(19.08 FPS)|143.34ms(6.98 FPS)|331.93ms(3.01 FPS)|1344.07ms(0.74 FPS)
S3FD|21.75ms(45.95 FPS)|55.73ms(17.94 FPS)|119.53ms(8.37 FPS)|471.31ms(2.21 FPS)
SSH|22.44ms(44.47 FPS)|55.29ms(18.09 FPS)|118.43ms(8.44 FPS)|463.10ms(2.16 FPS)
FaceBoxes3.2×|6.80ms(147.00 FPS)|12.96ms(77.19 FPS)|25.37ms(39.41 FPS)|111.98ms(8.93 FPS)
**LFFD**|7.60ms(131.40 FPS)|16.37ms(61.07 FPS)|31.27ms(31.98 FPS)|87.79ms(11.39 FPS)

Latency on NVIDIA TX2 (MXNet+CUDA 9.0+CUDNN7.1) presented in the paper:

Resolution->|160×120|320×240|640×480
------------|-------|--------|---------
FaceBoxes3.2×|11.20ms(89.29 FPS)|19.62ms(50.97 FPS)|72.74ms(13.75 FPS)
**LFFD**|7.30ms(136.99 FPS)|19.64ms(50.92 FPS)|64.70ms(15.46 FPS)

Latency on Respberry Pi 3 Model B+ (ncnn) presented in the paper:

Resolution->|160×120|320×240|640×480
------------|-------|--------|---------
FaceBoxes3.2×|167.20ms(5.98 FPS)|686.19ms(1.46 FPS)|3232.26ms(0.31 FPS)
**LFFD**|118.45ms(8.44 FPS)|409.19ms(2.44 FPS)|4114.15ms(0.24 FPS)

On NVIDIA platform, TensorRT is the best choice for inference. So, for additional NVIDIA hardwares, 
we use TensorRT 5.1.5+CUDA 10.0+CUDNN 7.4 for latency evaluation. As for ARM based platform, we plan 
to use MNN and Tengine for inference. We omit the evaluation for other methods here, LFFD only:

Resolution\\Platform |NVIDIA Jetson Nano |GTX 1060 (laptop) |RTX 2080TI
---------------------|-------------------|------------------|----------
160×120|xxx|xxx|xxx
640×480|xxx|xxx|xxx
1280×720|xxx|xxx|xxx
1920×1080|xxx|xxx|xxx
3840×2160|xxx|xxx|xxx

For ARM based platforms, all cores are used for inference:

Resolution\\Platform |Respberry Pi 3 Model B+ (MNN)|Respberry Pi 3 Model B+ (Tengine)|RK3288 (MNN)|RK3288 (Tengine)
---------------------|-------------------|------------------|----------|--------------
160×120|xxx|xxx|xxx|xxx
640×480|xxx|xxx|xxx|xxx


## Getting Started
We implement the proposed method using MXNet Module API.

#### Prerequirements
* Python>=3.5
* numpy>=1.16
* MXNet>=1.4 ([install guide](http://mxnet.incubator.apache.org/versions/master/install/index.html?platform=Linux&language=Python&processor=GPU))
* cv2=3.x (pip3 install opencv-python==3.4.5.20, other version should work as well)

> Tips: 
  * use MXNet with cudnn.
  * build numpy from source with OpenBLAS. This will improve the training efficiency.
  * make sure cv2 links to libjpeg-turbo, not libjpeg.

#### Advanced
1, float16 int8
2, pruning
3, branch cut
4, 

#### Tips for lower latency 

## Citation

## To Do List
