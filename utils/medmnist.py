import medmnist
from medmnist import INFO

def build_dataset(data_flag, mode, transforms = None, target_transforms = None, size = 224 ):
    assert 'mnist' in data_flag, f"Invalid Mode selected, {mode}"
    info  = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    dataset = DataClass( split=mode, transform=transforms, 
                        target_transform=target_transforms, download=True, size= size)
    return dataset

