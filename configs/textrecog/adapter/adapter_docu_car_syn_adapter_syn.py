_base_ = [
    "../_base_/datasets/syn.py",
    "../_base_/datasets/car.py",
    "../_base_/datasets/document.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_adam_base.py",
    "_base_adapter.py",
]

TASK_ID = 0
load_from = "work_dirs/backbone_docu_car_syn/epoch_20.pth"
work_dir = "work_dirs/backbone_docu_car_syn_adapter_syn"
train_mode = "adapter"
NUM_CLASSES = [11378, 11378]
NUM_TASKS = len(NUM_CLASSES)


optim_wrapper = dict(
    optimizer=dict(lr=1e-5, weight_decay=0.01), clip_grad=dict(max_norm=10, norm_type=2)
)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
train_cfg = dict(max_epochs=20, val_interval=1)
param_scheduler = [
    dict(type="LinearLR", end=2, start_factor=0.1, convert_to_iter_based=True),
    dict(type="MultiStepLR", milestones=[10, 15], end=20),
]

train_list = [_base_.syn_textrecog_train]
test_list = [_base_.syn_textrecog_test]

train_dataset = dict(
    type="ConcatDataset", datasets=train_list, pipeline=_base_.train_pipeline
)
test_dataset = dict(
    type="ConcatDataset", datasets=test_list, pipeline=_base_.test_pipeline
)

train_dataloader = dict(
    batch_size=256,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=train_dataset,
)

test_dataloader = dict(
    batch_size=32,
    num_workers=6,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=test_dataset,
)

val_dataloader = test_dataloader

val_evaluator = dict(dataset_prefixes=["syn"])
test_evaluator = val_evaluator

auto_scale_lr = dict(base_batch_size=256)


dictionary = dict(
    type="Dictionary",
    dict_file="{{ fileDirname }}/../../../dicts/chinese_english_digits.txt",
    with_start=True,
    with_end=True,
    same_start_end=True,
    with_padding=False,
    with_unknown=True,
)
model = dict(
    type="AdapterEncodeDecodeRecognizer",
    preprocessor=None,
    backbone=dict(
        type="AdaptersResnet",
        num_blocks=[2, 2, 2, 2],
        num_tasks=NUM_TASKS,
        num_classes=NUM_CLASSES,
    ),
    encoder=None,
    decoder=dict(
        type="AdapterTransformerDecoder",
        block_args=dict(
            input_dim=512,
            num_heads=8,
            dim_feedforward=128,
            adapter_dim=128,
            dropout=0.0,
        ),
        num_layers=2,
        num_classes=NUM_CLASSES,
        module_loss=dict(type="CTCModuleLoss", letter_case="lower", zero_infinity=True),
        postprocessor=dict(type="CTCPostProcessor"),
        dictionary=dictionary,
        task_id=TASK_ID,
    ),
    task_id=TASK_ID,
    data_preprocessor=dict(type="TextRecogDataPreprocessor", mean=[127], std=[127]),
)

find_unused_parameters = True
