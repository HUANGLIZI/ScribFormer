import numpy as np

np.set_printoptions(threshold=np.inf)

import network.scribformer_cam
class ScribFormer(network.scribformer_cam.Net):
    def __init__(self, linear_layer, bilinear, num_classes=4):
        super(ScribFormer, self).__init__(patch_size=16, in_chans=1, channel_ratio=4, embed_dim=384, depth=12,
                      num_heads=6, mlp_ratio=4, qkv_bias=True, drop_rate=0.0, drop_path_rate=0.1, num_classes=num_classes,
                                                           linear_layer=linear_layer, bilinear=bilinear)