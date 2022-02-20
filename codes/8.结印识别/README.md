# 结印识别

<img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/img20211201102837.png?x-oss-process=style/wp" style="width:200px;" />



我测试的环境：

Windows11、conda、Python3.8、CUDA（11.1）、Pytorch==1.8.0。



* Mediapipe：用于手部探测，但不做坐标获取（mediapipe双手交互时识别率较低）
* [InterHand2.6M](https://github.com/facebookresearch/InterHand2.6M): 用于手指3D坐标获取，模型文件较大，请去[这里下载](https://github.com/facebookresearch/InterHand2.6M/releases/tag/v1.0)
* Pytorch、[GCN](https://docs.dgl.ai/)：用于手印动作分类模型，我训练了6类动作（模型：saveModel下的handsModel.pth），泛化能力可能不强，请尽量自己训练。



项目主体在Demo文件夹下（其他文件夹都是InterHand2.6M clone来的）

* demo.py：采集训练数据和识别用；
* train.ipynb：训练数据用。





### 微信技术交流、问题反馈：

<img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/imgIMG_5862.JPG?x-oss-process=style/wp" style="width:200px;" />

