
syn_data_root = 'data/Meta-SelfLearning/LMDB/syn/'

syn_textrecog_train = dict(
    type='RecogLMDBDataset',
    data_root=syn_data_root,
    ann_file='train_imgs.lmdb',
    pipeline=None)

syn_textrecog_test = dict(
    type='RecogLMDBDataset',
    data_root=syn_data_root,
    ann_file='test_imgs.lmdb',
    test_mode=True,
    pipeline=None)


