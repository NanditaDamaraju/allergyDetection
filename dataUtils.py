
import pandas as pd
import os
from PIL import Image 
from preprocessUtils import *

# 

def load_data_from_folder_into_dataframe(img_dir):
	filenames = os.listdir(img_dir) # read in the names of the image files 
	filenames = [x.split('.')[0] for x in filenames] # remove the .jpg extensions for joining later 
	img_files_df = pd.DataFrame(filenames, columns = ['Individual Patch']) # create a pandas dataframe 
	img_files_df['full_file_path'] = img_dir  + img_files_df['Individual Patch']  + '.jpg'
	return img_files_df

def load_binary_outcomes_from_csv(labels_csv_path):
	labels_df = pd.read_csv(labels_csv_path, usecols=['Individual Patch', 'Binary Outcome'])
	return labels_df

def merge_image_with_labels(img_dir, labels_csv_path, img_w=100,  img_h=100, img_ch = 3):
	
	img_files_df = load_data_from_folder_into_dataframe(img_dir)
	labels_df = load_binary_outcomes_from_csv(labels_csv_path)

	full_data_points = img_files_df.merge(labels_df, on='Individual Patch')
	num_imgs, _ = full_data_points.shape

	imgs = np.zeros((img_w, img_h, img_ch, num_imgs))
	labels = np.zeros(num_imgs)

	for i, img_file in enumerate(full_data_points['full_file_path'].values): 
    
	    # display progress
	    if i % 100 == 0:
	        print(i, 'out of', num_imgs, 'images processed')
	        
	    # read in the image     
	    img = np.asarray(Image.open(img_file))
	    
	    # run pre-processing 
	    imgs[:,:,:,i] = preprocess(img)
	    
	    labels[i] = full_data_points['Binary Outcome'].iloc[i] 

	return imgs, labels

def load_data_from_folder_into_matrix(img_dir,	img_w=100,  img_h=100, img_ch = 3):
	
	img_files_df = load_data_from_folder_into_dataframe(img_dir)
	num_imgs, _ = img_files_df.shape

	imgs = np.zeros((img_w, img_h, img_ch, num_imgs))

	for i, img_file in enumerate(img_files_df['full_file_path'].values): 
    
	    # display progress
	    if i % 100 == 0:
	        print(i, 'out of', num_imgs, 'images processed')
	        
	    # read in the image     
	    img = np.asarray(Image.open(img_file))
	    
	    # run pre-processing 
	    imgs[:,:,:,i] = preprocess(img)
	    

	return imgs
