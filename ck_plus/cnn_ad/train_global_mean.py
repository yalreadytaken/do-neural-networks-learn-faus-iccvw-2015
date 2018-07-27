import argparse
import os
import sys
sys.path.append('..')

import numpy
import random

from anna import util
from anna.datasets import supervised_dataset
#from anna.datasets.supervised_data_loader import SupervisedDataLoaderCrossVal

import data_fold_loader
import data_paths
from model import SupervisedModel

def hide_patch(img):
    # get width and height of the image
    s = img.shape
    wd = s[1]
    ht = s[2]
 
    # randomly choose one grid size
    patch_size = patch_sizes[random.randint(0,len(patch_sizes)-1)]

    # hide the patches
    if (patch_size != 0):
        for x in range(0,wd,patch_size):
            for y in range(0,ht,patch_size):
                x_end = min(wd, x+patch_size)  
                y_end = min(ht, y+patch_size)
                if(random.random() <= hide_prob):
                    img[:,x:x_end,y:y_end] = 0

    return img

parser = argparse.ArgumentParser(prog='train_cnn_with_dropout_\
                                      data_augmentation',
                                 description='Script to train convolutional \
                                 network from random initialization with \
                                 dropout and data augmentation.')
parser.add_argument("-s", "--split", default='0', help='Testing split of CK+ \
                    to use. (0-9)')
parser.add_argument("--hide_prob", default='0.5', help='By what probability \
                    patches will be hidden.')
parser.add_argument('--patch_size', metavar='N', type=int, nargs='+',
                    help='Possible patch size, 0 means no hiding.')
parser.add_argument("--checkpoint_dir", default='./', help='Location to save \
                    model checkpoint files.')
args = parser.parse_args()

# hiding probability
hide_prob = args.hide_prob

# possible patch size, 0 means no hiding
patch_sizes = args.patch_size


print('Start')
test_split = int(args.split)
if test_split < 0 or test_split > 9:
    raise Exception("Testing Split must be in range 0-9.")
print('Using CK+ testing split: {}'.format(test_split))

checkpoint_dir = os.path.join(args.checkpoint_dir, 'checkpoints_'+str(test_split))
print 'Checkpoint dir: ', checkpoint_dir

pid = os.getpid()
print('PID: {}'.format(pid))
f = open('pid_'+str(test_split), 'wb')
f.write(str(pid)+'\n')
f.close()

# Load model
model = SupervisedModel('experiment', './', learning_rate=1e-2)
monitor = util.Monitor(model,
                       checkpoint_directory=checkpoint_dir,
                       save_steps=1000)
#util.load_checkpoint(model, "./checkpoints_2/experiment-07m-23d-05h-26m-49s.pkl")
# Add dropout to fully-connected layer
model.fc4.dropout = 0.5
model._compile()

# Loading CK+ dataset
print('Loading Data')
#supervised_data_loader = SupervisedDataLoaderCrossVal(
#    data_paths.ck_plus_data_path)
#train_data_container = supervised_data_loader.load('train', train_split)
#test_data_container = supervised_data_loader.load('test', train_split)

train_folds, val_fold, _ = data_fold_loader.load_fold_assignment(test_fold=test_split)
X_train, y_train = data_fold_loader.load_folds(data_paths.ck_plus_data_path, train_folds)
X_val, y_val = data_fold_loader.load_folds(data_paths.ck_plus_data_path, [val_fold])
X_test, y_test = data_fold_loader.load_folds(data_paths.ck_plus_data_path, [test_split])

X_train = numpy.float32(X_train)
X_train /= 255.0
X_train *= 2.0

X_val = numpy.float32(X_val)
X_val /= 255.0
X_val *= 2.0

X_test = numpy.float32(X_test)
X_test /= 255.0
X_test *= 2.0

mean = numpy.average(numpy.concatenate((X_train, X_val, X_test), axis=0), axis=(0,2,3))
std = numpy.std(numpy.concatenate((X_train, X_val, X_test), axis=0), axis=(0,2,3))
print mean
print std

train_dataset = supervised_dataset.SupervisedDataset(X_train, y_train)
val_dataset = supervised_dataset.SupervisedDataset(X_val, y_val)
train_iterator = train_dataset.iterator(
    mode='random_uniform', batch_size=64, num_batches=31000)
val_iterator = val_dataset.iterator(
    mode='random_uniform', batch_size=64, num_batches=31000)


# Do data augmentation (crops, flips, rotations, scales, intensity)
data_augmenter = util.DataAugmenter2(crop_shape=(96, 96),
                                     flip=True)#, gray_on=True)
normer = util.Normer3(filter_size=5, num_channels=1)
module_list_train = [data_augmenter]#, normer]
module_list_val = [normer]
preprocessor_train = util.Preprocessor(module_list_train)
preprocessor_val = util.Preprocessor(module_list_val)

print('Training Model')
for x_batch, y_batch in train_iterator:
    x_batch = (x_batch - mean) / std
    x_batch = preprocessor_train.run(x_batch)
    # loop over batch
    for i in range(len(x_batch)):
        # hide patch for an image
        x_batch[i] = hide_patch(x_batch[i])

    monitor.start()
    log_prob, accuracy = model.train(x_batch, y_batch)
    monitor.stop(1-accuracy)

    if monitor.test:
        monitor.start()
        x_val_batch, y_val_batch = val_iterator.next()
        #x_val_batch = preprocessor_val.run(x_val_batch)
        x_val_batch = (x_val_batch - mean) / std
        val_accuracy = model.eval(x_val_batch, y_val_batch)
        monitor.stop_test(1-val_accuracy)
