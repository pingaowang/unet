from data import *

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

myGenerator = get_train_generator(20,'data/membrane/train','image','label',data_gen_args,
                                  save_to_dir = "data/membrane/train/aug")

num_batch = 3
for i,batch in enumerate(myGenerator):
    if(i >= num_batch):
        break

## Save NPY data
"""
If your computer has enough memory, you can create npy files containing all your images and masks, and feed your DNN with them.
"""
# image_arr,mask_arr = geneTrainNpy("data/membrane/train/aug/","data/membrane/train/aug/")
# np.save("data/image_arr.npy",image_arr)
# np.save("data/mask_arr.npy",mask_arr)