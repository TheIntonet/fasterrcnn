# fasterrcnn
faster R-CNN in Keras and Tensorflow 2<br>
<h1>What is in here?</h1>
- simple to understand the code<br>
- RPN (region proposal layer) can be trained separately!<br>
- VGG16 support<br>
- prediction script included<br>
- own dataset :)<br> 

<h1>Frameworks</h1><br>
- Tensorflow==2.2.0<br>
- Keras==2.3.1

<h1>Running scripts..</h1>
Make sure you've installed the requirements!<br><br>
<pre>
<code>pip install -r requirements.txt</code>
</pre>

<h1>3. lets train region proposal network first, rather than training the whole network.</h1>
It's better to train first the RPN, it is much handlier.<br><br>
<pre>
<code>python train_rpn.py -p data.txt </code>
</pre>
