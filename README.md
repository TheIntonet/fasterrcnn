# fasterrcnn
faster R-CNN in Keras and Tensorflow 2<br>
<h1>What is in here?</h1>
<ul>
  <li>simple to understand the code</li>
  <li>RPN (region proposal layer) can be trained separately!</li>
  <li>VGG16 support</li>
  <li>prediction script included</li>
  <li>own dataset :)</li>
</ul>

<h1>Frameworks</h1><br>
<ul>
  <li>Tensorflow==2.2.0</li>
  <li>Keras==2.3.1</li>
</ul>

<h1>1. Running scripts..</h1>
Make sure you've installed the requirements!<br><br>
<pre>
<code>pip install -r requirements.txt</code>
</pre>

<h1>2. lets train region proposal network first, rather than training the whole network.</h1>
It's better to train first the RPN, it is much handlier.<br><br>
<pre>
<code>python train_rpn.py -p data.txt </code>
</pre>
