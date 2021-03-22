data_source_cfg = dict(type='ImageList', memcached=False, mclient_path=None)
data_root = "/home/cjrd/data/smth_smth/object_crops/all"
data_all_list = "/home/cjrd/data/smth_smth/smth_smth_object_list.txt"
split_at = [500000000]
split_name = ['smthsmth_all']

# TODO
img_norm_cfg = dict(mean=[0.4686, 0.4211, 0.3930], std=[0.2588, 0.2538, 0.2563])

data = dict(
    imgs_per_gpu=32,
    workers_per_gpu=2,
    extract=dict(
        type="ExtractDataset",
        data_source=dict(
            list_file=data_all_list, root=data_root, **data_source_cfg),
        pipeline=[
            dict(type='Resize', size=256),
            dict(type='Resize', size=(224, 224)),
            dict(type='ToTensor'),
            dict(type='Normalize', **img_norm_cfg),
        ]))
