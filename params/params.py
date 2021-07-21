#K, path_to_chkpt, path_to_backup, path_to_Wi, batch_size, path_to_preprocess, frame_shape, path_to_mp4

#number of frames to load
K = 8

#path to main weight
path_to_chkpt = 'model_weights.tar'

#path to backup
path_to_backup = 'backup_model_weights.tar'

#CHANGE first part
path_to_Wi = ""+"UBC_Fashion_Wi_weights"
# path_to_Wi = "test/"+"Wi_weights"

#CHANGE if better gpu
batch_size = 8

#dataset save path
path_to_preprocess = '/home/ubuntu/playpen/Realistic-Neural-Talking-Head-Models/data/preprocess'

#default for Voxceleb
frame_shape = 256

#path to dataset
path_to_mp4 = '/home/ubuntu/playpen/Realistic-Neural-Talking-Head-Models/data/mp4'
