
document_data_root = 'data/Meta-SelfLearning/LMDB/document/'

document_textrecog_train = dict(
    type='RecogLMDBDataset',
    data_root=document_data_root,
    ann_file='train_imgs.lmdb',
    pipeline=None)

document_textrecog_test = dict(
    type='RecogLMDBDataset',
    data_root=document_data_root,
    ann_file='test_imgs.lmdb',
    test_mode=True,
    pipeline=None)


