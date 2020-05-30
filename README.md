# fasterrcnn
faster R-CNN in Keras and Tensorflow 2<br>
# What is in here?
<ul>
  <li>simple to understand the code</li>
  <li>RPN (region proposal layer) can be trained separately!</li>
  <li>VGG16 support</li>
  <li>prediction script included</li>
  <li>own dataset :)</li>
</ul>

# Frameworks<br>
<ul>
  <li>Tensorflow==2.2.0</li>
  <li>Keras==2.3.1</li>
</ul>

# 1. Running scripts..
Make sure you've installed the requirements!<br><br>
<pre>
<code>pip install -r requirements.txt</code>
</pre>

# 2. lets train region proposal network first, rather than training the whole network.
It's better to train first the RPN, it is much handlier.<br><br>
<pre>
<code>
#required arguments
python train_rpn.py -p data.txt

#all arguments
python train_rpn.py -p [path_to_annotation_file] -n [num_rois] --hf [horizontal_flips] -vf [vertical_flips] --rot [rot_90] --num_epochs [epochs] --config_filename [path_to_config_file] --elen [epoch_length] --source [source_of_the_weight_file] --output_weight_path [output_weight_path] --input_weight_path [rpn_input_weight]

Epoch 1/50
10/10 [==============================] - 183s 18s/step - loss: 0.9712 - rpn_out_class_loss: 0.9433 - rpn_out_regress_loss: 0.0279 - val_loss: 4.7013e-06 - val_rpn_out_class_loss: 0.5662 - val_rpn_out_regress_loss: 0.0170
Epoch 2/50
10/10 [==============================] - 188s 19s/step - loss: 1.0685 - rpn_out_class_loss: 1.0535 - rpn_out_regress_loss: 0.0151 - val_loss: 3.0161e-04 - val_rpn_out_class_loss: 0.6133 - val_rpn_out_regress_loss: 0.0183
Epoch 3/50
10/10 [==============================] - 158s 16s/step - loss: 0.9711 - rpn_out_class_loss: 0.9599 - rpn_out_regress_loss: 0.0112 - val_loss: 3.3153e-05 - val_rpn_out_class_loss: 0.5662 - val_rpn_out_regress_loss: 0.0167
Epoch 4/50
 6/10 [=================>............] - ETA: 16s - loss: 1.6198 - rpn_out_class_loss: 1.5726 - rpn_out_regress_loss: 0.0471
</code>
</pre>

# 3. train the whole Faster-RCNN network!
After you have trained your RPN, you can now train the whole network.
<pre>
<code>
#required arguments
python train_frcnn.py -p data.txt

#all arguments
python train_frcnn.py -p [path_to_annotation_file] -n [num_rois] --hf [horizontal_flips] -vf [vertical_flips] --rot [rot_90] --num_epochs [epochs] --config_filename [path_to_config_file] --elen [epoch_length] --output_weight_path [output_weight_path] --source [source_of_the_weight_file]  --input_weight_path [rpn_input_weight] --rpn [rpn_weight_path] --opt [optimizers] --elen [epoch_length] --load [path_to_frcnn_model] --cat [category_to_train_on] --lr [learn_rate]

Epoch 1/4
10/10 [==============================] - 94s 9s/step - rpn_cls: 0.2273 - rpn_regr: 0.0063 - detector_cls: 0.0781 - detector_regr: 0.0825 - average number of objects: 1.2000
Mean number of bounding boxes from RPN overlapping ground truth boxes: 1.2
Classifier accuracy for bounding boxes from RPN: 0.9499999940395355
Loss RPN classifier: 0.2691028459822974
Loss RPN regression: 0.009440940673084697
Loss Detector classifier: 0.1330762797035277
Loss Detector regression: 0.08800292604137212
Elapsed time: 94.21539258956909
Epoch 2/4
Average number of overlapping bounding boxes from RPN = 1.2 for 10 previous iterations
10/10 [==============================] - 95s 9s/step - rpn_cls: 2.2784 - rpn_regr: 0.0369 - detector_cls: 0.5874 - detector_regr: 0.2051 - average number of objects: 1.3000
Mean number of bounding boxes from RPN overlapping ground truth boxes: 1.3
Classifier accuracy for bounding boxes from RPN: 0.8899999856948853
Loss RPN classifier: 1.6048255547648995
Loss RPN regression: 0.017651171068428083
Loss Detector classifier: 0.3393111705780029
Loss Detector regression: 0.2202323233243078
Elapsed time: 94.70645260810852
Epoch 3/4
Average number of overlapping bounding boxes from RPN = 1.3 for 10 previous iterations
 6/10 [=================>............] - ETA: 38s - rpn_cls: 1.1460 - rpn_regr: 0.0317 - detector_cls: 0.2123 - detector_regr: 0.2767 - average number of objects: 1.6667
</code>
</pre>

# 4. predict on your images
To predict your images you have to put them all in a folder and the script shows you the prediction.
<pre>
<code>
#required arguments:
python predict.py -p [path_to_the_image_directory]

#all aruments:
python predict.py -p [path_to_the_image_directory] -n [num_rois] --config_filename [path_to_config_file] --write [save_the_prediction_result] --load [path_to_model]
</code>
</pre>

# Dataset setup
First you copy all you data images into the dataset/images folder. Then you run the script 'resize_img.py' to resize all your images to the optimal size(224, 224, 3) they will get to the dataset\destimges folder. Next you label them all with the <a href="https://github.com/tzutalin/labelImg">labelimg</a> tool and safe the xml files in the dataset/data folder. After that you go back and run the script 'xmltoannotation.py' which create from the xml files one annotation file called data.txt. This file you can copy to the main directory to have simpler access.
This is the final format:
<pre>
<code>
filepath,x1,y1,x2,y2,class_name
</code>
</pre>
