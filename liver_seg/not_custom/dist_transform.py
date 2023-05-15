from monai.config import KeysCollection
from monai.transforms import MapTransform

from boundaryloss.dataloader import dist_map_transform

class DistTransformd(MapTransform):

    def __init__(self, keys: KeysCollection):
        super().__init__(keys)
        
        self.disttransform = dist_map_transform([1, 1, 1, 1], 2) #pixdim 1, 1, 1; 3 classes

    def __call__(self, data):
        d = dict(data)
        
        d["dist_map"] = self.disttransform(d["label"])
        return d
