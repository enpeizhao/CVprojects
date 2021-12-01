# Copyright (c) Facebook, Inc. and its affiliates.

import json
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl

# load camera positions
campos_dict = {}
for split in ('train', 'test', 'val'):
    with open('../../data/InterHand2.6M/annotations/' + split + '/InterHand2.6M_' + split + '_camera.json') as f:
        cameras = json.load(f)
    for capture_id in cameras.keys():
        for cam_id in cameras[capture_id]['campos'].keys():
            campos = np.array(cameras[capture_id]['campos'][cam_id], dtype=np.float32)

            # exact camera positions can be slightly different for each 'capture_id'.
            # however, just overwrite them for the visualization purpose.
            if cam_id not in campos_dict:
                campos_dict[cam_id] = campos

# plot camera positions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
cmap = plt.get_cmap('rainbow')
cam_num = len(campos_dict)
colors = [cmap(i) for i in np.linspace(0, 1, cam_num)]
colors = [np.array((c[2], c[1], c[0])) for c in colors]

for i, k in enumerate(campos_dict.keys()):
    ax.scatter(campos_dict[k][0], campos_dict[k][2], -campos_dict[k][1], c=colors[i], marker='o')
    ax.text(campos_dict[k][0], campos_dict[k][2], -campos_dict[k][1], k)

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
plt.show()

