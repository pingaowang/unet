from model import *
from data import *

# config
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
MAX_EPOCH = 10
TRAIN_PATH = 'data/membrane/train'
TEST_PATH = 'data/membrane/test_2'

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
                    steps_per_epoch=300,
                    epochs=MAX_EPOCH,
                    callbacks=[model_checkpoint])

# test
testGene = testGenerator(TEST_PATH)
results = model.predict_generator(testGene, 30, verbose=1)
saveResult(TEST_PATH, results)