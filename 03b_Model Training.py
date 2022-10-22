# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/computer-vision-foundations. For more information about this solution accelerator, visit https://www.databricks.com/blog/2021/12/17/enabling-computer-vision-applications-with-the-data-lakehouse.html.

# COMMAND ----------

# MAGIC %md The purpose of this notebook is to demonstrate patterns supporting computer vision model training.  
# MAGIC 
# MAGIC * If you intend to deploy the model as a microservice or as part of a Spark pipeline using user-defined functions, use the previous **03a** notebook , which uses one of the latest versions of Databricks runtime.  
# MAGIC * If you intend to deploy the model to an edge device, you should use a cluster where the Python version used by the cluster is aligned with the version deployed on your device.  Our Raspberry Pi device runs Python 3.7, and for that reason, we have trained our edge-deployed models in this notebook **03b** on the Databricks 7.3 ML cluster which runs that same version of Python.

# COMMAND ----------

# DBTITLE 1,Verify Python Version
import sys

print('You are running a Databricks {0} cluster leveraging Python {1}'.format( 
  spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion"),
  sys.version.split(' ')[0])
  )

# COMMAND ----------

# DBTITLE 1,Retrieve Configurations
# MAGIC %run "./01_Configuration"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from petastorm import TransformSpec

from PIL import Image

import torchvision
import torch

from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK
import horovod.torch as hvd
from sparkdl import HorovodRunner

import mlflow

import pyspark.sql.functions as f

import numpy as np
from functools import partial
import io

# COMMAND ----------

# MAGIC %md ## Introduction
# MAGIC 
# MAGIC This notebook will demonstrate how we train a computer vision model at scale.  The general workflow we will follow will take data from the table we loaded with image data in our last notebook and then load it into a high-performance, scalable caching layer.  Leveraging that cache, we will then generate numerous model iterations in order to discover an optimal set of hyperparameters for our model.  Those optimized hyperparameter values will then be used to train a final model in a scalable, distributed manner for later deployment.
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/cv_model_training_workflow.png' width=600>

# COMMAND ----------

# MAGIC %md ## Step 1: Access Data
# MAGIC 
# MAGIC Our image data resides in a Delta Table named *images*.  The raw binary for each image is available through a field named *content*. The relatively small size of each image along with the small overall number of images in our table would allow us to extract our data to a pandas dataframe against which we could then train our model.  However, we'd like to establish a pattern that would allow our processing to scale, and for that, we'll retrieve our data to a [Petastorm](https://docs.databricks.com/applications/machine-learning/load-data/petastorm.html) cache.
# MAGIC 
# MAGIC Petastorm is a technology available in the Databricks platform which allows us to retrieve data using the distributed power of the cluster and then cache it to Parquet files for fast access by Tensorflow, PyTorch and PySpark.  The caching of data in this manner allows us to train models on volumes of data that might otherwise overwhelm the memory resources available on an individual cluster node.
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/cv_petastorm.png' width=500>
# MAGIC 
# MAGIC Our first step in building the Petastorm cache is to retrieve the image data and divide it into training and testing sets:

# COMMAND ----------

# DBTITLE 1,Retrieve Train & Test Images
# retrieve images of interest
images = (
  spark
    .table(config['input_images_table'])
    .select('content', 'label', 'path') # path will be used as a unique identifier in next steps
  )

# retrieve stratified sample of images
images_train = images.sampleBy('label', fractions={0: 0.8, 1: 0.8}) # 80% sample from each class to training
images_test = images.join(images_train, on='path', how='leftanti') # remainder to testing

# drop any unnecessary fields
images_train = images_train.drop('path').repartition(sc.defaultParallelism) # drop path identifier
images_test = images_test.drop('path').repartition(sc.defaultParallelism)

# verify sampling
display(
  images_train
    .withColumn('eval_set', f.lit('train'))
    .union( images_test.withColumn('eval_set', f.lit('test')))
    .groupBy('eval_set', 'label')
      .agg(f.count('*').alias('instances'))
    .orderBy('eval_set', 'label')
  )

# COMMAND ----------

# MAGIC %md With the sets defined, we persist them with Petastorm using a Spark (to Petastorm) converter.  This should be a relatively simple task but we ran into an interesting problem when using Horovod (Step 4 in this notebook) with the Petastorm cache which we will explain here to help others who may encounter a similar issue.
# MAGIC 
# MAGIC Petastorm writes its output to Parquet files.  Each Parquet file consists of one or more internal row groups. Horovod expects the total number of row groups available to it to be equal to or greater than the number of shards (parallel processes) it employs.  In Step 4, we direct Horovod to leverage a single GPU per cluster worker or the toatl number of virtual CPUs across cluster workers for its shards.  We must therefore ensure Petastorm generates enough row groups to align with that number.
# MAGIC 
# MAGIC To do this, we consider that each row in our dataset consists of a binary image of variable size and an 4-byte integer label. We sum the bytes associated with each for each of the training and testing datasets and divide that number by the number of virtual CPUs (as presented by *sc.defaultParallelism*) to ensure we align with Horovod's requirements:

# COMMAND ----------

# DBTITLE 1,Clean Up Old Petastorm Cache (If Exists)
try:
  dbutils.fs.rm(config['petastorm_path'],True)
except:
  pass

# COMMAND ----------

# DBTITLE 1,Create Petastorm Cache
# configure destination for petastore cache
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, config['petastorm_path'])

# determine rough bytes in dataset
bytes_in_train = images_train.withColumn('bytes', f.lit(4) + f.length('content')).groupBy().agg(f.sum('bytes').alias('bytes')).collect()[0]['bytes']
bytes_in_test = images_test.withColumn('bytes', f.lit(4) + f.length('content')).groupBy().agg(f.sum('bytes').alias('bytes')).collect()[0]['bytes']

# cache images data
converter_train = make_spark_converter(images_train, parquet_row_group_size_bytes=int(bytes_in_train/sc.defaultParallelism))
converter_test = make_spark_converter(images_test, parquet_row_group_size_bytes=int(bytes_in_test/sc.defaultParallelism))

# COMMAND ----------

# MAGIC %md With the data cached, we now need to consider how we will access it.  The instructions for reading data from the cache are captured in a Petastorm [TransformSpec](https://petastorm.readthedocs.io/en/latest/api.html#module-petastorm.transform). The spec defines how records are not only read from the cache but the transformations required before the data can be made available to the consuming model. This can include image preprocessing transformations.
# MAGIC 
# MAGIC Our TransformSpec will read raw images from the cache and return a resized, normalized tensor (along with image labels) as expected by the PyTorch model we will employ.  An important part of the spec definition is identifying the shape of the tensor object returned with each image. As we will be taking our 600x600 3-channel (RGB) images down to a 256x256 image size (while preserving the RGB channels), our spec is defined as follows.  Please note that that the configuration of the *Normalize* transformation is [specified](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/) by the TorchVision model we'll eventually employ:

# COMMAND ----------

# DBTITLE 1,Define TransformSpec
# define logic for the transformation of images
def transform_row(is_train, batch_pd):
  
  # transform images
  # -----------------------------------------------------------
  # transform step 1: read incoming content value as an image
  transformers = [torchvision.transforms.Lambda(lambda x: Image.open(io.BytesIO(x)))]
  
  # transform step 2: resize, normalize and transform to tensor
  transformers.extend([
      torchvision.transforms.Resize(224), # resize to 224 x 224 image
      torchvision.transforms.ToTensor(), # convert to torch.FloatTensor
      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
  
  # assemble transformation steps into a pipeline
  trans = torchvision.transforms.Compose(transformers)
  
  # apply pipeline to images 
  batch_pd['features'] = batch_pd['content'].map(lambda x: trans(x).numpy())
  # -----------------------------------------------------------
  
  # transform labels (our evaluation metric expects values to be float32)
  # -----------------------------------------------------------
  batch_pd['label'] = batch_pd['label'].astype('float32')
  # -----------------------------------------------------------
  
  return batch_pd[['features', 'label']]
 
# define function to retrieve transformation spec
def get_transform_spec(is_train=True):
  
  spec = TransformSpec(
            partial(transform_row, is_train), # function to call to retrieve/transform row
            edit_fields=[  # schema of rows returned by function
                ('features', np.float32, (3, 224, 224), False), 
                ('label', np.float32, (), False)
                ], 
            selected_fields=['features', 'label'] # fields in schema to send to model
            )
  
  return spec

# COMMAND ----------

# MAGIC %md With the cache and transform spec in place, we can perform a quick test to see that everything is working and to get a sense of the data coming through:

# COMMAND ----------

# DBTITLE 1,Test TransformSpec
# access petastorm cache and transform data using spec
with converter_train.make_torch_dataloader(
      transform_spec=get_transform_spec(is_train=True), 
      batch_size=1
      ) as train_dataloader:

  # retrieve records from cache
  for i in iter(train_dataloader):
      print(i)
      break

# COMMAND ----------

# MAGIC %md ## Step 2: Define Model
# MAGIC 
# MAGIC With data access defined, we will turn our attention to model training.  Our first step is simply to get our model defined and to verify it reads data from Petastorm correctly. Once we have this working, we can tackle the distribution of model training cycles.
# MAGIC 
# MAGIC To keep things simple, we'll leverage  the [MobileNetV2 model](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/#model-description) available through TorchVision which has been pre-trained on the [ImageNet dataset](https://www.image-net.org/about.php). Our model will attempt to differentiate between two image classes, *i.e.* those that contain an object of interest and those that do not.  It will read images 32 at a time.  It will take 5 passes over the entire training dataset:
# MAGIC 
# MAGIC **NOTE** We are borrowing heavily from [this notebook](https://docs.databricks.com/_static/notebooks/deep-learning/petastorm-spark-converter-pytorch.html) and have attempted to preserve the codes structure to provide an easier cross-reference whenever possible. In addition, we are not diving into the topics of image classification or the use of pre-trained models for computer vision scenarios so that we may remain focused on model training mechanics.  These other topics are ones we'll tackle in future notebooks.

# COMMAND ----------

# DBTITLE 1,Control Parameters
NUM_CLASSES = 2  # two classes in labels (0 or 1)
BATCH_SIZE = 32  # process 32 images at a time
NUM_EPOCHS = 5   # iterate over all images 5 times

# COMMAND ----------

# DBTITLE 1,Get the Model MobileNetV2 from Torchvision
def get_model():
  
  # access pretrained model
  model = torchvision.models.mobilenet_v2(pretrained=True)
  
  # freeze parameters in the feature extraction layers
  for param in model.parameters():
    param.requires_grad = False
    
  # add a new classifier layer for transfer learning
  num_ftrs = model.classifier[1].in_features
  
  # parameters of newly constructed modules have requires_grad=True by default
  model.classifier[1] = torch.nn.Linear(num_ftrs, NUM_CLASSES)
  
  return model

# COMMAND ----------

# MAGIC %md To train and evaluate our model, we'll define some additional helper functions.  Notice the *train_one_epoch* function iterates over the Petastorm cache, retrieving data through the *train_dataloader_iter* variable, an item that will defined in subsequent steps:

# COMMAND ----------

# DBTITLE 1,Define the Train & Evaluate Function for the Model
def train_one_epoch(
  model,
  criterion,
  optimizer,
  scheduler, 
  train_dataloader_iter,
  steps_per_epoch,
  epoch, 
  device
  ):
  
  model.train()  # set model to training mode
 
  # statistics
  running_loss = 0.0
  running_corrects = 0
 
  # iterate over the data for one epoch.
  for step in range(steps_per_epoch):
    
    # retrieve next batch from petastorm
    pd_batch = next(train_dataloader_iter)
    
    # seperate input features and labels
    inputs, labels = pd_batch['features'].to(device), pd_batch['label'].to(device)
    
    # track history in training
    with torch.set_grad_enabled(True):
      
      # zero the parameter gradients
      optimizer.zero_grad()
 
      # forward
      outputs = model(inputs)
      _, preds = torch.max(outputs, 1)
      probs = torch.nn.functional.softmax(outputs, dim=0)[:,1]
      loss = criterion(probs, labels)
 
      # backward + optimize
      loss.backward()
      optimizer.step()
 
    # statistics
    running_loss += loss.item() * inputs.size(0)
    running_corrects += torch.sum(preds == labels.data)
  
  scheduler.step()
 
  epoch_loss = running_loss / (steps_per_epoch * BATCH_SIZE)
  epoch_acc = running_corrects.double() / (steps_per_epoch * BATCH_SIZE)
 
  print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
  return epoch_loss, epoch_acc

 
def evaluate(
  model, 
  criterion, 
  test_dataloader_iter,
  test_steps, 
  device, 
  metric_agg_fn=None
  ):
  
  model.eval()  # set model to evaluate mode
 
  # statistics
  running_loss = 0.0
  running_corrects = 0
 
  # iterate over all the validation data.
  for step in range(test_steps):
    
    pd_batch = next(test_dataloader_iter)
    inputs, labels = pd_batch['features'].to(device), pd_batch['label'].to(device)
 
    # do not track history in evaluation to save memory
    with torch.set_grad_enabled(False):
      
      # forward
      outputs = model(inputs)
      _, preds = torch.max(outputs, 1)
      probs = torch.nn.functional.softmax(outputs, dim=1)[:,1]
      loss = criterion(probs, labels)
 
    # statistics
    running_loss += loss.item()
    running_corrects += torch.sum(preds == labels.data)
  
  # average the losses across observations for each minibatch.
  epoch_loss = running_loss / test_steps
  epoch_acc = running_corrects.double() / (test_steps * BATCH_SIZE)
  
  # metric_agg_fn is used in the distributed training to aggregate the metrics on all workers
  if metric_agg_fn is not None:
    epoch_loss = metric_agg_fn(epoch_loss, 'avg_loss')
    epoch_acc = metric_agg_fn(epoch_acc, 'avg_acc')
 
  print('Testing Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
  return epoch_loss, epoch_acc

# COMMAND ----------

# MAGIC %md With everything in place, we might perform a simple model training and evaluation operation.  This will take place on a single node in our cluster, but if it works, we'll scale this to run in a distributed manner:

# COMMAND ----------

# DBTITLE 1,Train & Evaluate the Model on a Single Node
def train_and_evaluate(lr=0.001, momentum=0.0):
  
  # determine if gpu available for compute
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  # get model
  model = get_model()
  
  # assign model to process on identified processor device
  model = model.to(device)
 
  # optimize for binary cross entropy
  criterion = torch.nn.BCELoss()
 
  # only parameters of final layer are being optimized.
  optimizer = torch.optim.SGD(model.classifier[1].parameters(), lr=lr, momentum=momentum)
 
  # decay LR by a factor of 0.1 every 7 epochs
  exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
  
  # access data in petastorm cache
  with converter_train.make_torch_dataloader(transform_spec=get_transform_spec(is_train=True), 
                                             batch_size=BATCH_SIZE) as train_dataloader, \
       converter_test.make_torch_dataloader(transform_spec=get_transform_spec(is_train=False), 
                                           batch_size=BATCH_SIZE) as val_dataloader:
    
    # define iterator for data access and number of cycles required
    train_dataloader_iter = iter(train_dataloader)
    steps_per_epoch = len(converter_train) // BATCH_SIZE
    
    val_dataloader_iter = iter(val_dataloader)
    validation_steps = max(1, len(converter_test) // BATCH_SIZE)
    
    # for each epoch 
    for epoch in range(NUM_EPOCHS):
      
      print('Epoch {}/{}'.format(epoch + 1, NUM_EPOCHS))
      print('-' * 10)
 
      # train
      train_loss, train_acc = train_one_epoch(model, criterion, optimizer, exp_lr_scheduler, 
                                              train_dataloader_iter, steps_per_epoch, epoch, 
                                              device)
      # evaluate
      val_loss, val_acc = evaluate(model, criterion, val_dataloader_iter, validation_steps, device)
 
  # correct a type issue with acc
  if type(val_acc)==torch.Tensor: val_acc = val_acc.item() 
    
  return val_loss, val_acc # extract value from tensor

loss, acc = train_and_evaluate(**{'lr':0.01, 'momentum':0.9})

# COMMAND ----------

# MAGIC %md ##Step 3: Perform Hyperparameter Tuning
# MAGIC 
# MAGIC Now that we've demonstrated our ability to train the model, let's focus on hyperparameter tuning.  Our model, like most models, have a number of parameters that control its behavior during training.  The setting of these parameters is challenging in that there are a wide range of potential values for each parameter and complex interactions between variables when applied against a given data set.   
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/cv_hyperopt.png' width=500>
# MAGIC 
# MAGIC A common way of dealing with this complexity is to train a series of models using different hyperparameter values to determine which combinations produce the best results. Using [Hyperopt](https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/index.html#hyperparameter-tuning-with-hyperopt), we can intelligently navigate a range of hyperparameter values to efficiently arrive at an optimal configuration.  This often requires us to train hundreds or even thousands of models to discover an optimum. Leveraging the *SparkTrials* feature, we can tackle these training iterations across the resources provided by our cluster, allowing us to shorten the time for this work based on the number of resources we wish to provision:

# COMMAND ----------

# DBTITLE 1,Perform Hyperparameter Tuning with Hyperopt
# define hyperparameter search space
search_space = {
    'lr': hp.loguniform('lr', -10, -4),
    'momentum': hp.loguniform('momentum', -10, 0)
    }


# define training function to return results as expected by hyperopt
def train_fn(params):
  
  # train model against a provided set of hyperparameter values
  loss, acc = train_and_evaluate(**params)
  
  # log this iteration to mlflow for greater transparency
  mlflow.log_metric('accuracy', acc)
  
  # return results from this iteration
  return {'loss': loss, 'status': STATUS_OK}


# determine degree of parallelism to employ
if torch.cuda.is_available(): # is gpu
  parallelism = int(sc.getConf().get('spark.databricks.clusterUsageTags.clusterWorkers'))
else: # is cpu
  parallelism = sc.defaultParallelism

# perform distributed hyperparameter tuning
with mlflow.start_run(run_name=config['tuning_model_name']) as run:
  
  argmin = fmin(
    fn=train_fn,
    space=search_space,
    algo=tpe.suggest,
    max_evals=10, # total number of hyperparameter runs (this would typically be much higher)
    trials=SparkTrials(parallelism=parallelism)) # number of hyperparameter runs to run in parallel
  

# COMMAND ----------

# DBTITLE 1,Display Optimized Hyperparameter Values
# See optimized hyperparameters
argmin

# COMMAND ----------

# MAGIC %md Please note in the hyperparameter run, we leveraged [mlflow](https://mlflow.org/), another technology pre-integrated with the Databricks ML runtime, to capture various metrics for our runs.  With hyperopt, tracking is automatic but by explicitly calling to mlflow, we can log additional metrics such as accuracy to help us explore hyperparameter tuning.
# MAGIC 
# MAGIC With access to these data, we can use the mlflow [experiment interface](https://docs.databricks.com/applications/mlflow/tracking.html) and explore how various hyperparameters impact model metrics:</p>
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/cv_model_hyperparam.PNG' width=90%>

# COMMAND ----------

# MAGIC %md ##Step 4: Train Optimized Model
# MAGIC 
# MAGIC With optimal hyperparameters identified, we can now train an optimized version of our model.  While we have demonstrated we could do this on a single node, we might take advantage of [Horovod](https://docs.databricks.com/applications/machine-learning/train-model/distributed-training/horovod-runner.html) to distribute this work across the nodes in our cluster.  In a nutshell, Horovod provides a framework for partitioning the training of a Tensorflow or PyTorch model and aggregating the results to form a final, consolidated model.
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/cv_horovod.png' width=500>
# MAGIC 
# MAGIC While conceptually simple, the use of Horovod does require we us to rewrite our core helper functions in order to distribute the work. Per the [Horovod documentation for PyTorch](https://horovod.readthedocs.io/en/stable/pytorch.html), the approach requires us to:</p>
# MAGIC 
# MAGIC 1. Initialize Horovod
# MAGIC 2. Align the Horovod processes to specific CPU cores or GPUs
# MAGIC 3. Scale the learning rate based on the number of Horovod processes
# MAGIC 4. Wrap the model optimizer for distribution
# MAGIC 5. Initialize state variables associated with the Horovod processes
# MAGIC 
# MAGIC Once configured, Horovod will send a separate batch of data to each CPU or GPU enlisted in the training exercise. The scaling of the learning rate is intended to compensate for the fact that each training node is not seeing the full set of the data.  The collective learning is "averaged" with each epoch through the calling of an *allreduce* method at the time of model evaluation: 

# COMMAND ----------

# DBTITLE 1,Define Train & Evaluate Function for Horovod
# define function for model evaluation
def metric_average_hvd(val, name):
  tensor = torch.tensor(val)
  avg_tensor = hvd.allreduce(tensor, name=name)
  return avg_tensor.item()


# define function for distributed training & evaluation
def train_and_evaluate_hvd(lr=0.001, momentum=0.0):
  
  # Step 1: Initialize Horovod
  hvd.init()
  
  # Step 2: Align the Horovod processes to specific CPU cores or GPUs
  
  # determine devices to use for training
  if torch.cuda.is_available():  # gpu
    torch.cuda.set_device(hvd.local_rank())
    device = torch.cuda.current_device()
  else:
    device = torch.device("cpu") # cpu
  
  # retrieve model and associate with device
  model = get_model()
  model = model.to(device)  
  criterion = torch.nn.BCELoss()
  
  # Step 3: Scale the learning rate based on the number of Horovod processes
  optimizer = torch.optim.SGD(model.classifier[1].parameters(), lr=lr * hvd.size(), momentum=momentum)
  
  # Step 4: Wrap the optimizer for distribution
  optimizer_hvd = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
  exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_hvd, step_size=7, gamma=0.1)
  
  # Step 5: Initialize state variables associated with the Horovod processes
  hvd.broadcast_parameters(model.state_dict(), root_rank=0)
  hvd.broadcast_optimizer_state(optimizer, root_rank=0)
  
  # open up access to the petastorm cache
  with converter_train.make_torch_dataloader(
          transform_spec=get_transform_spec(is_train=True), 
          cur_shard=hvd.rank(), 
          shard_count=hvd.size(),
          batch_size=BATCH_SIZE
          ) as train_dataloader, \
       converter_test.make_torch_dataloader(
          transform_spec=get_transform_spec(is_train=False),
          cur_shard=hvd.rank(), 
          shard_count=hvd.size(),
          batch_size=BATCH_SIZE
          ) as test_dataloader:
    
    # each core/gpu will handle a batch
    train_dataloader_iter = iter(train_dataloader)
    train_steps = len(converter_train) // (BATCH_SIZE * hvd.size())
    test_dataloader_iter = iter(test_dataloader)
    test_steps = max(1, len(converter_test) // (BATCH_SIZE * hvd.size()))
    
    # iterate over dataset
    for epoch in range(NUM_EPOCHS):
      
      # print epoch info
      print('Epoch {}/{}'.format(epoch + 1, NUM_EPOCHS))
      print('-' * 10)
 
      # train model
      train_loss, train_acc = train_one_epoch(
                              model, 
                              criterion, 
                              optimizer_hvd, 
                              exp_lr_scheduler, 
                              train_dataloader_iter, 
                              train_steps, 
                              epoch, 
                              device
                              )
    
      # evaluate model
      test_loss, test_acc = evaluate(
                              model, 
                              criterion, 
                              test_dataloader_iter, 
                              test_steps,
                              device, 
                              metric_agg_fn=metric_average_hvd
                              )

  return test_loss, test_acc, model

# COMMAND ----------

# MAGIC %md Now we can train our model.  We'll log the model parameters, evaluation metrics and the model itself in mlflow as part of this process:

# COMMAND ----------

# DBTITLE 1,Perform Distributed Training
# determine parallelism available to horovod
if torch.cuda.is_available(): # is gpu
  parallelism = int(sc.getConf().get('spark.databricks.clusterUsageTags.clusterWorkers'))
else:
  parallelism = 2 # setting the parallelism low at 2 for the small data size; otherwise feel free to set to sc.defaultParallelism

# initialize runtime environment for horovod
hr = HorovodRunner(np=parallelism) 

# run distributed training
with mlflow.start_run(run_name=config['tuned_model_name_73']) as run:
  
  # train and evaluate model
  loss, acc, model = hr.run(train_and_evaluate_hvd, **argmin) # argmin contains tuned hyperparameters
  
  # log model in mlflow
  mlflow.log_params(argmin)
  mlflow.log_metrics({'loss':loss, 'accuracy':acc})
  mlflow.pytorch.log_model(model,'model')

# COMMAND ----------

# MAGIC %md ## Step 5: Drop Petastorm Cache
# MAGIC 
# MAGIC Now that we're done training the model, we can drop the cached data.  It's important we do this step in order to ensure any subsequent training runs do not see redundant information in the cache folders from prior iterations:

# COMMAND ----------

# DBTITLE 1,Drop Cache
converter_train.delete()
converter_test.delete()

# COMMAND ----------

# MAGIC %md © 2021 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
