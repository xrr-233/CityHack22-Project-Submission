## Techniques

### Back-End Model

1. Feature Engineering

   Each sample of train data provide only 5 values[time, x1, x2, x3, x4], this do not consider the time-series information. To solve this problem, we create the feature by ourselves. The first feature is the same time last 7 days. (for exmaple, the time is 01-30 11:10, then the feature are 01-23 11:10,  01-24 11:10, ... ,01-29 11:10). The second feature is the time catogory. (00:00 as 0, 00:10 as 1, ... , 23:50 as 143). The third feature is the X1 to X4 in last two hours and the fourth feature is the demand in last two hours.

2. Preprocessing

   As the predicted value is demand, we use the log scale as we care about the percentage of difference. We also normalize the first, third, fourth feature using this formula (x-mean)/(max-min) and convert time catogory to one-hot vector.

3. Training

   We use cosineAnnealing as scheduler, adam as optimizor, early stop, drop-out and weight decay to avoid overfitting. We split the training dataset to train/test/val for our evaluation and training for 35 epoches.

4. Ensembling

   For the final prediction, we ensemble the autoML model and the home-made model to achieve better performance.

### Front-end Website

1. Flask Framework

   Flask is applied in our project to give a better experience to the users. It utilizes the functions of werkzurg and jinja and allows other extensions to develop the website much more easily. It contains the structure of a static folder (for storing images and other miscellaneous files), a template folder (for rendering HTML files), and several back-end functions which allow front-end and back-end interaction.

2. Bootstrap

   It is a good plugin to make the composing of the web pages more decent.

3. Ajax

   A convenient tool for asynchronous data transmission.

4. Highcharts

   The Highcharts library contains many tools needed to create fancy and reliable data visualizations.
