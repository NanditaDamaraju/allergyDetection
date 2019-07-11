import numpy as np

def downsample_dataset(imgs, labels):
	negative_indices = np.where(labels == 0)[0]
	positive_indices = np.where(labels == 1)[0]
	# let's even out the dataset 
	negative_downsample_indices = np.random.choice(negative_indices, len(positive_indices), replace=False)
	downsample_indices = np.concatenate((positive_indices, negative_downsample_indices))

	downsampled_train_imgs = imgs[:,:,:,downsample_indices]
	downsampled_train_imgs = downsampled_train_imgs[:,:]
	downsampled_train_labels = labels[downsample_indices] 

	return downsampled_train_imgs, downsampled_train_labels

def augment_dataset(imgs, labels):
# Perform data augmentations to increase the size of the dataset 

	r, c, ch, num_imgs = imgs.shape

	flip_ud = np.flip(imgs, 0)
	flip_lr = np.flip(imgs, 1)
	rot90 = np.rot90(imgs, 1)
	rot180 = np.rot90(imgs, 2)
	rot270 = np.rot90(imgs, 3)

	# Put them all together in one matrix 
	aug_imgs = np.concatenate((imgs, flip_ud, flip_lr, rot90, rot180, rot270), axis=3)
	aug_labels = np.tile(labels, 6) 

	# Verify the shape of the resulting matrix 
	print("Shape of Augmented dataset", aug_imgs.shape)
	print("Shape of Augmented labels", aug_labels.shape)

	return aug_imgs, aug_labels

def augment_with_random_sampling(imgs, labels):
	
	positive_indices = np.where(labels == 1)[0]
	negative_indices = np.where(labels == 0)[0]

	positive_imgs = imgs[:,:,:,positive_indices]

	flip_ud = np.flip(positive_imgs, 0)
	flip_lr = np.flip(positive_imgs, 1)
	rot90 = np.rot90(positive_imgs, 1)
	rot180 = np.rot90(positive_imgs, 2)
	rot270 = np.rot90(positive_imgs, 3)

	aug_positive_imgs = np.concatenate((positive_imgs, flip_ud, flip_lr, rot90, rot180, rot270), axis=3)
	aug_positive_labels = np.ones((aug_positive_imgs.shape[3]))

	negative_downsample_indices = np.random.choice(negative_indices, len(positive_indices)*6, replace=False)
	negative_downsample_imgs = imgs[:,:,:, negative_downsample_indices]
	negative_downsample_labels = np.zeros(negative_downsample_imgs.shape[3])
	
	aug_imgs = np.concatenate((aug_positive_imgs, negative_downsample_imgs), axis=3)
	aug_labels = np.concatenate((aug_positive_labels, negative_downsample_labels))

	print("Shape of Augmented dataset", aug_imgs.shape)
	print("Shape of Augmented labels", aug_labels.shape)

	return aug_imgs, aug_labels





	
