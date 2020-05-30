from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import os

import keras
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model, load_model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
from keras_frcnn.simple_parser import get_data
import keras_frcnn.roi_ as roi_helpers
from keras.utils import generic_utils
import xml.etree.ElementTree as ET

# gpu setting
#if 'tensorflow' == K.backend():
#    import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
#config2 = tf.compat.v1.ConfigProto()
#config2.gpu_options.allow_growth = True
#tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config2))

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="train_path", help="Path to training data.")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois", help="Number of RoIs to process at once.", default=10)
parser.add_option("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=true).", action="store_false", default=True)
parser.add_option("--vf", dest="vertical_flips", help="Augment with vertical flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--rot", "--rot_90", dest="rot_90", help="Augment with 90 degree rotations in training. (Default=false).",
				  action="store_true", default=False)
parser.add_option("--num_epochs", type="int", dest="num_epochs", help="Number of epochs.", default=50)
parser.add_option("--config_filename", dest="config_filename", help=
				"Location to store all the metadata related to the training (to be used when testing).",
				default="config.pickle")
parser.add_option("--output_weight_path", dest="output_weight_path", help="Output path for weights.", default='./model_frcnn.hdf5')
parser.add_option("--source", dest="source", help="Set the source of the weights.(all/rpn)", default='all')
parser.add_option("--input_weight_path", dest="input_weight_path", help="Input path for weights. If not specified, will try to load default weights provided by keras.", default=None)
parser.add_option("--rpn", dest="rpn_weight_path", help="Input path for rpn.", default=None)
parser.add_option("--opt", dest="optimizers", help="set the optimizer to use", default="SGD")
parser.add_option("--elen", dest="epoch_length", help="set the epoch length. def=1000", default=1000)
parser.add_option("--load", dest="load", help="What model to load", default=None)
parser.add_option("--dataset", dest="dataset", help="name of the dataset", default="voc")
parser.add_option("--cat", dest="cat", help="categroy to train on. default train on all cats.", default=None)
parser.add_option("--lr", dest="lr", help="learn rate", type=float, default=1e-3)

(options, args) = parser.parse_args()

if not options.train_path:   # if filename is not given
    parser.error('Error: path to training data must be specified. Pass --path to command line')

# pass the settings from the command line, and persist them in the config object
C = config.Config()
model_path = C.model_path

#xml
tree = ET.parse('weights.xml')
root = tree.getroot()
best_loss = root[0][3].text
best_loss = float(best_loss)

C.use_horizontal_flips = bool(options.horizontal_flips)
C.use_vertical_flips = bool(options.vertical_flips)
C.rot_90 = bool(options.rot_90)

# mkdir to save models.
if not os.path.isdir("models"):
  os.mkdir("models")

C.num_rois = int(options.num_rois)

# we will use resnet. may change to others
from keras_frcnn import vgg16 as nn

# check if weight path was passed via command line
if options.input_weight_path:
    C.model_path = options.input_weight_path
else:
    # set the path to weights based on backend and model
    C.model_path = nn.get_weight_path()

all_imgs, classes_count, class_mapping = get_data(options.train_path, options.cat)

if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping

inv_map = {v: k for k, v in class_mapping.items()}

print('Training images per class:')
pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))

config_output_filename = options.config_filename

with open(config_output_filename, 'wb') as config_f:
    pickle.dump(C,config_f)
    print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))

random.shuffle(all_imgs)

num_imgs = len(all_imgs)

train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))


data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length, K.common.image_dim_ordering(), mode='train')
data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, nn.get_img_output_length,K.common.image_dim_ordering(), mode='val')

if K.common.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
else:
    input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.base_vgg(img_input) # trainable=True

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)

from keras_frcnn import rpn
rpn = rpn.rpn(shared_layers, num_anchors)

from keras_frcnn import classifier
classifier = classifier.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count)) #, trainable=True

model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)

# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

if options.load is not None:
    C.model_path = options.load
elif options.rpn_weight_path is not None:
    C.model_path = options.rpn_weight_path
elif options.source == 'rpn':
	print("oin Moin")
	rpn_path = root[0][0].text
	C.model_path = rpn_path
elif options.source == 'all':
	all_path = root[0][2].text
	C.model_path = all_path

# load pretrained weights
try:
	print('loading weights from {}'.format(C.model_path))
	model_rpn.load_weights(C.model_path, by_name=True)
	try:
		model_classifier.load_weights(C.model_path, by_name=True)
	except Exception as e:
		print("Classifier weights couldn't be loaded...")
except:
    print('Could not load pretrained model weights.')

# optimizer setup
if options.optimizers == "SGD":
    if options.rpn_weight_path is not None:
        optimizer = SGD(lr=options.lr/100, decay=0.0005, momentum=0.9)
        optimizer_classifier = SGD(lr=options.lr/5, decay=0.0005, momentum=0.9)
    else:
        optimizer = SGD(lr=options.lr/10, decay=0.0005, momentum=0.9)
        optimizer_classifier = SGD(lr=options.lr/10, decay=0.0005, momentum=0.9)
else:
    optimizer = Adam(lr=options.lr, clipnorm=0.001)
    optimizer_classifier = Adam(lr=options.lr, clipnorm=0.001)

# compile the model AFTER loading weights!
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')

epoch_length = int(options.epoch_length)
num_epochs = int(options.num_epochs)
iter_num = 0

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

#best_loss = np.Inf

class_mapping_inv = {v: k for k, v in class_mapping.items()}
print('Starting training')

vis = True

for epoch_num in range(num_epochs):
	progbar = generic_utils.Progbar(epoch_length)
	print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

	# first 3 epoch is warmup
	if epoch_num == 3 and options.rpn_weight_path is not None:
		K.set_value(model_rpn.optimizer.lr, options.lr/30)
		K.set_value(model_classifier.optimizer.lr, options.lr/3)

	while True:
		try:
			if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
			    mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
			    rpn_accuracy_rpn_monitor = []
			    print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
			    if mean_overlapping_bboxes == 0:
			      print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')
			X, Y, img_data = next(data_gen_train)

			loss_rpn = model_rpn.train_on_batch(X, Y)

			P_rpn = model_rpn.predict_on_batch(X)
			R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.common.image_dim_ordering(), use_regr=True, overlap_thresh=0.4, max_boxes=300)
			# note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
			X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)

			if X2 is None:
			    rpn_accuracy_rpn_monitor.append(0)
			    rpn_accuracy_for_epoch.append(0)
			    continue

			neg_samples = np.where(Y1[0, :, -1] == 1)
			pos_samples = np.where(Y1[0, :, -1] == 0)

			if len(neg_samples) > 0:
			    neg_samples = neg_samples[0]
			else:
			    neg_samples = []

			if len(pos_samples) > 0:
			    pos_samples = pos_samples[0]
			else:
			    pos_samples = []

			rpn_accuracy_rpn_monitor.append(len(pos_samples))
			rpn_accuracy_for_epoch.append((len(pos_samples)))

			if C.num_rois > 1:
			    if len(pos_samples) < C.num_rois//2:
                                selected_pos_samples = pos_samples.tolist()
			    else:
                                selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
			    try:
                                selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
			    except:
                                selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()
			    sel_samples = selected_pos_samples + selected_neg_samples
			else:
			    # in the extreme case where num_rois = 1, we pick a random pos or neg sample
			    selected_pos_samples = pos_samples.tolist()
			    selected_neg_samples = neg_samples.tolist()
			    if np.random.randint(0, 2):
                                sel_samples = random.choice(neg_samples)
			    else:
                                sel_samples = random.choice(pos_samples)

			loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

			losses[iter_num, 0] = loss_rpn[1]
			losses[iter_num, 1] = loss_rpn[2]

			losses[iter_num, 2] = loss_class[1]
			losses[iter_num, 3] = loss_class[2]
			losses[iter_num, 4] = loss_class[3]

			iter_num += 1

			progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
									  ('detector_cls', np.mean(losses[:iter_num, 2])), ('detector_regr', np.mean(losses[:iter_num, 3])),
                                     ("average number of objects", len(selected_pos_samples))])

			if iter_num == epoch_length:
				loss_rpn_cls = np.mean(losses[:, 0])
				loss_rpn_regr = np.mean(losses[:, 1])
				loss_class_cls = np.mean(losses[:, 2])
				loss_class_regr = np.mean(losses[:, 3])
				class_acc = np.mean(losses[:, 4])

				mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
				rpn_accuracy_for_epoch = []

				if C.verbose:
					print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
					print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
					print('Loss RPN classifier: {}'.format(loss_rpn_cls))
					print('Loss RPN regression: {}'.format(loss_rpn_regr))
					print('Loss Detector classifier: {}'.format(loss_class_cls))
					print('Loss Detector regression: {}'.format(loss_class_regr))
					print('Elapsed time: {}'.format(time.time() - start_time))

				curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
				iter_num = 0
				start_time = time.time()

				if curr_loss < best_loss:
					if C.verbose:
						print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
					best_loss = curr_loss
					model_all.save_weights(model_path)
					root[0][2].text = model_path
					root[0][3].text = str(best_loss)
					print("Saved " + model_path)
					tree.write('weights.xml')
				break

		except Exception as e:
			print('Exception: {}'.format(e))
			continue

print('Training complete.')
