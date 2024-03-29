from .otb import OTBDataset
from .uav import UAVDataset
from .lasot import LaSOTDataset
from .got10k import GOT10kDataset
from .nfs import NFSDataset
from .vot import VOTDataset, VOTLTDataset
from .trackingnet import TrackingNetDataset


class DatasetFactory(object):
    @staticmethod
    def create_dataset(**kwargs):
        """
        Args:
            name: dataset name 'OTB2015', 'LaSOT', 'UAV123', 'NFS240', 'NFS30',
                'VOT2018', 'VOT2016', 'VOT2018-LT'
            dataset_root: dataset root
            load_img: wether to load image
        Return:
            dataset
        """
        assert 'name' in kwargs, "should provide dataset name"
        name = kwargs['name']
        if 'OTB100' in name:
            dataset = OTBDataset(**kwargs)
        elif 'LaSOT' == name:
            dataset = LaSOTDataset(**kwargs)
        elif 'UAV123' in name:
            dataset = UAVDataset(**kwargs)
        elif 'GOT' == name:
            dataset = GOT10kDataset(**kwargs)
        elif 'NFS30' == name or 'NFS240' == name:
            dataset = NFSDataset(**kwargs)
        elif 'VOT2018' == name or 'VOT2016' == name or 'VOT2019' == name:
            dataset = VOTDataset(**kwargs)
        elif 'VOT2018-LT' == name:
            dataset = VOTLTDataset(**kwargs)
        elif 'TrackingNet' == name:
            dataset = TrackingNetDataset(**kwargs)
        else:
            raise Exception("unknow dataset {}".format(kwargs['name']))
        return dataset

