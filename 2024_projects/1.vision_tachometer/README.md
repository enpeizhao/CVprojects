## Vision Tachometer

> If you have any questions, please submit issues or email me: enpeicv@outlook.com, have fun with it!

### 1.Demo


https://github.com/enpeizhao/CVprojects/assets/6524256/f2f0f5c7-c75d-4db4-9593-a182d5604dcb



### 2.Usage

* Clone the project

**Python**

* Create a `python=3.8` environment with `conda` or `venv`.

* Install packages like `opencv` and `numpy`.

* Download `1.video_tachometer_demo.mp4` from [media files](https://github.com/enpeizhao/CVprojects/releases/tag/media).

* Move `1.video_tachometer_demo.mp4` to the `media` directory of the project.

* ```bash
  # video_path: filename for input video | camera index for stream
  # snap_path: reference snapshot
  # output_path: recorded video path
  
  # example
  python demo.py --video_path media/1.video_tachometer_demo.mp4 --snap_path media/snap.png --output_path result.mp4
  ```

* It will open three windows (raw video, matched, graph).

  ![](https://enpei-md.oss-cn-hangzhou.aliyuncs.com/202401131140446.png?x-oss-process=style/wp)

  

**C++**

> The overall process is identical to Python's.

* Build and run:

* ```bash
  # build
  cmake -S . -B build
  cmake --build build
  
  # run 
  ./build/HelloWorld <video_file> <reference_image>
  ```

### 3.References

* [High Speed Rotation Estimation with Dynamic Vision Sensors](https://arxiv.org/pdf/2209.02205.pdf)
* https://ieeexplore.ieee.org/document/8443343
* https://kar.kent.ac.uk/77844/1/1570319863_I2MTC2017_speed%20measurement_final.pdf
* https://www.mdpi.com/1424-8220/20/24/7314
* https://www.semanticscholar.org/reader/d5cc5ae8e75ef87ac8824ae13c7a4d3d11e92be8
* https://doi.org/10.1364/OE.24.013375
* https://linkinghub.elsevier.com/retrieve/pii/S088832701730537X

