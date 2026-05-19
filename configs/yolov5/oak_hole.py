_base_ = 'yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'

data_root = '/home/xyzt2204/project/holedet/oak/oak_datasetv1/'
class_name = ('hole', )
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(20, 220, 60)])

# anchors = [
#     [(20, 20), (154, 91), (143, 162)],  # P3/8
#     [(242, 160), (189, 287), (391, 207)],  # P4/16
#     [(353, 337), (539, 341), (443, 432)]  # P5/32
# ]

anchors = [
    [(20, 20), (23, 18), (22, 21)],  # P3/8
    [(25, 20), (23, 24), (26, 23)],  # P4/16
    [(25, 26), (32, 28), (42, 38)]  # P5/32
]

max_epochs = 600
train_batch_size_per_gpu = 24
train_num_workers = 4

load_from = '/home/xyzt2204/project/mmyolo/mmyolo/work_dirs/oak_hole/epoch_30.pth'  # noqa

model = dict(
    backbone=dict(frozen_stages=0),
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
        prior_generator=dict(base_sizes=anchors)))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/trainval.json',
        data_prefix=dict(img='images/')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/')))

test_dataloader = val_dataloader

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

val_evaluator = dict(ann_file=data_root + 'annotations/test.json')
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=10, save_best='auto'),
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),
    logger=dict(type='LoggerHook', interval=5),
    visualization=dict(
        draw=True,
        test_out_dir='./vis_res/',
        type='mmdet.DetVisualizationHook'))
train_cfg = dict(max_epochs=max_epochs, val_interval=10)
# visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')]) # noqa
