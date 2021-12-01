import dgl
from dgl.data import DGLDataset
import os
# from dgl import save_graphs, load_graphs
# from dgl.data.utils import makedirs, save_info, load_info
import glob
import json
import numpy as np 
import torch as th

class HandsData(DGLDataset):
    """ 用于在DGL中自定义图数据集的模板：

    Parameters
    ----------
    url : str
        下载原始数据集的url。
    raw_dir : str
        指定下载数据的存储目录或已下载数据的存储目录。默认: ~/.dgl/
    save_dir : str
        处理完成的数据集的保存目录。默认：raw_dir指定的值
    force_reload : bool
        是否重新导入数据集。默认：False
    verbose : bool
        是否打印进度信息。
    """
    def __init__(self,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False):
        
        self.label_class  = ['0','1','2','3','4','5','6']
        
        super(HandsData, self).__init__(name='dataset_name',
                                        url=url,
                                        raw_dir=raw_dir,
                                        save_dir=save_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)

    def download(self):
        # 将原始数据下载到本地磁盘
        pass

    def process(self):
        # 将原始数据处理为图、标签和数据集划分的掩码
        # mat_path = self.raw_path + '.mat'
        # 将数据处理为图列表和标签列表
        self.graphs, self.label = self._load_graph()

    def _load_graph(self):

        # 从json中获取文件
        # label_class = ['0','1','2','3','4',]
        graphs = []
        labels = []

        for label in self.label_class:
            json_dir = './trainingData/'+label+'/*'
            
            json_files = glob.glob(json_dir)

            for json_item in json_files:

                # Opening JSON file
                f = open(json_item)
                
                # a dictionary
                json_data = json.load(f)
                
                # Iterating through the json
                # list
                

                # 构造图以及特征
                u,v = th.tensor([[0,0,0,0,0,4,3,2,8,7,6,12,11,10,16,15,14,20,19,18,0,21,21,21,21,21,25,24,23,29,28,27,33,32,31,37,36,35,41,40,39],
                 [4,8,12,16,20,3,2,1,7,6,5,11,10,9,15,14,13,19,18,17,21,25,29,33,37,41,24,23,22,28,27,26,32,31,30,36,35,34,40,39,38]])
                g = dgl.graph((u,v))
                
                # 无向处理
                bg = dgl.to_bidirected(g)
                
                # x，y，z坐标
                x_y_z_column = self.relativeMiddleCor(json_data)

                # 添加特征
                bg.ndata['feat'] =th.tensor( x_y_z_column ) # x,y,z坐标

                graphs.append(bg)
                labels.append(int(label)-3) # 如果从非0 标签开始训练，这里需要减去对应数字

                # print('处理1条JSON')
                
                # Closing file
                f.close()

        return graphs, th.tensor(labels)


    def relativeMiddleCor(self,data):
        # 计算相对于几何中心的坐标
        x_list = data['x_list']
        y_list = data['y_list']
        z_list = data['z_list']

        # 计算几何中心坐标
        min_x = min(x_list)
        max_x = max(x_list)

        min_y = min(y_list)
        max_y = max(y_list)

        min_z = min(z_list)
        max_z = max(z_list)

        middle_p_x = min_x+ 0.5*(max_x-min_x)
        middle_p_y = min_y+ 0.5*(max_y-min_y)
        middle_p_z = min_z+ 0.5*(max_z-min_z)


        # p(相对) = (x原始 -  Px(重心), y原始 -  Py(重心))
        x_list = np.array(x_list) - middle_p_x
        y_list = np.array(y_list) - middle_p_y
        z_list = np.array(z_list) - middle_p_z

        x_y_z_column = np.column_stack((x_list, y_list,z_list))

        return x_y_z_column




    def __getitem__(self, idx):
        """ 通过idx获取对应的图和标签

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        (dgl.DGLGraph, Tensor)
        """
        return self.graphs[idx], self.label[idx]

    def __len__(self):
        """数据集中图的数量"""
        return len(self.graphs)

    def save(self):
        # 将处理后的数据保存至 `self.save_path`
        # graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        # save_graphs(str(graph_path), self.graphs, {'labels': self.label})
        pass

    def load(self):
        # 从目录 `self.save_path` 里读取处理过的数据
        # graphs, label_dict = load_graphs(os.path.join(self.save_path, 'dgl_graph.bin'))
        # self.graphs = graphs
        # self.label = label_dict['labels']
        pass

    def has_cache(self):
        # 检查在 `self.save_path` 中是否存有处理后的数据
        # graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        # return os.path.exists(graph_path)
        pass

    
    @property
    def num_labels(self):
        """每个图的标签数，即预测任务数。"""
        return len(self.label_class)

    def hello(self):
        print('hellow world')