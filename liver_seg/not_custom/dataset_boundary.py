import collections
from typing import Callable, Optional, Sequence, Union

from monai.data import Dataset
from monai.transforms import apply_transform

from torch.utils.data import Subset

from boundaryloss.dataloader import dist_map_transform

class BoundSet(Dataset):
    def __init__(self, data: Sequence, transform: Optional[Callable] = None) -> None:
        #self.filename: list[str] # Get the list as done usually
        #self.data = data
        #self.dataset_root: Path # Path to root of data
        super().__init__(data=data, transform=transform)

        self.disttransform = dist_map_transform([1, 1, 1], 3) #1, 1, 1 pixdim, 3 classes

    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        if isinstance(index, slice):
            # dataset[:42]
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            # dataset[[1, 3, 4]]
            return Subset(dataset=self, indices = index)
        return self._transform(index)

    def _transform(self, index: int):
        data_i = self.data[index]
        
