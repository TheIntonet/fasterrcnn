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

# train the whole Faster-RCNN network!
After you have trained your RPN, you can now train the whole network.
<pre>
<code>
#required arguments
python train_frcnn.py -p data.txt

#all arguments
python train_frcnn.py -p [path_to_annotation_file] -n [num_rois] --hf [horizontal_flips] -vf [vertical_flips] --rot [rot_90] --num_epochs [epochs] --config_filename [path_to_config_file] --elen [epoch_length] --output_weight_path [output_weight_path] --source [source_of_the_weight_file]  --input_weight_path [rpn_input_weight] --rpn [rpn_weight_path] --opt [optimizers] --elen [epoch_length] --load [path_to_frcnn_model] --cat [category_to_train_on] --lr [learn_rate]
</code>
</pre>
