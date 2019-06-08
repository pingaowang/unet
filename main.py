from model import *
from data import *
import os.path
from shutil import copyfile

# config
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
MAX_EPOCH = 1
STEP_PER_EPOCH = 3
TRAIN_PATH = 'data/membrane/train'
TEST_PATH = 'data/membrane/test_2'

if not os.path.isfile(TEST_PATH):
    os.mkdir(TEST_PATH)
    for i in range(29):
        copyfile(os.path.join('data/membrane/test', str(i) + '.png'),
                 os.path.join(TEST_PATH, str(i) + '.png'))


# Augmentation
data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')
myGene = get_train_generator(2, TRAIN_PATH, 'image', 'label', data_gen_args, save_to_dir=None)

# model
model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)

# fit
model.fit_generator(myGene,
                    steps_per_epoch=STEP_PER_EPOCH,
                    epochs=MAX_EPOCH,
                    callbacks=[model_checkpoint])

# test
testGene = testGenerator(TEST_PATH)
results = model.predict_generator(testGene, 30, verbose=1)
saveResult(TEST_PATH, results)