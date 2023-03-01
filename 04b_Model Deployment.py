# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/computer-vision-foundations. For more information about this solution accelerator, visit https://www.databricks.com/blog/2021/12/17/enabling-computer-vision-applications-with-the-data-lakehouse.html.

# COMMAND ----------

# MAGIC %md The purpose of this notebook is to demonstrate patterns for computer vision model deployment.  This notebook was developed on a **Databricks 7.3 ML LTS** cluster and follows notebook **03b**.  

# COMMAND ----------

# MAGIC %md ## Introduction
# MAGIC 
# MAGIC In notebook 04a and 04b, we explore three different deployment vehicles:</p>
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/cv_deploy4.png' width=550>
# MAGIC 
# MAGIC Each deployment path is facilitated by mlflow, a model management technology integrated with the Databricks workspace.  For the edge device deployment, you need to ensure the training environment is aligned with the Python version supported by the device and that you do not use any mlflow functionality, such as the model registry, that may impose updated requirements for the device.  For our Raspberry Pi device, which supports Python 3.7, we will make use of the Databricks 7.3 ML LTS runtime and avoid using the mlflow registry which *pickles* model components using pickle libraries aligned with latest versions of Python.

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
import mlflow

import torchvision
import torch

import PIL

import pyspark.sql.functions as f

import io, requests
import pandas as pd
import base64

# COMMAND ----------

# MAGIC %md ## Step 1: Persist Model with Transformation Logic
# MAGIC 
# MAGIC Having successfully trained our model, we can persist it in preparation for deployment.  If you examine the last cell of the last step of the last notebook, you'll see we actually did this when we called the mlflow *log_model* method.  However, that version of the model doesn't include the logic required to transform a raw image into the format expected by our trained model.  So, we'll take a moment here to write a wrapper for our model which tackles all the data transformations originally captured in our transform spec.  To clarify, we could have (and *should* have) defined this wrapper in the last notebook and logged the model to the mlflow register at that time.  However, we elected to tackle this here as part of our focus on deployment.
# MAGIC 
# MAGIC The custom wrapper for our model defines logic for two key methods: *\_\_init\_\_* and *predict*.  The *\_\_init\_\_* method defines the logic employed as the model is initialized.  It's here that we will flip our model into its evaluation mode and define the logic for data transformation.  The *predict* method receives input data, applies the transformations and returns scored output.  
# MAGIC 
# MAGIC The Spark user-defined function associated with our ETL deployment pattern passes data to the *predict* function as a pandas dataframe. To keep things simple, we'll package our image data in a similar manner before passing it to the model for scoring.  However, we could have included logic in our predict function to inspect the incoming data and respond to different data structures and formats:

# COMMAND ----------

# DBTITLE 1,Define Wrapper for CV Model
class CVModelWrapper(mlflow.pyfunc.PythonModel):
  
  def __init__(self, model):
    
    import torchvision
    from PIL import Image
    import io
    
    # instantiate model in evaluation mode
    self.model = model.eval()
    
     # define transformation pipeline
    trans = torchvision.transforms.Compose([
              torchvision.transforms.Lambda(lambda x: Image.open(io.BytesIO(x))),
              torchvision.transforms.Resize(256),
              torchvision.transforms.ToTensor(),
              torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
              ])
    self.transform = trans
    
  def predict(self, context, model_input):
    
    import pandas as pd
    import torch
    import base64
    
    # clean up data input
    if type(model_input.iloc[0]['content'])==str:  # if the content field contains strings (instead of byte arrays) ...
      model_input['content'] = model_input['content'].apply(lambda s: base64.b64decode(s)) # assume it has been base64 encoded & decode it to byte array
    
    # transform input images
    model_input['features'] = model_input['content'].map(lambda x: self.transform(x).numpy())
    
    # make predictions
    outputs = []
    for i in torch.utils.data.DataLoader(model_input['features']):
      o = model(i)      
      probs = torch.nn.functional.softmax(o, dim=1)[:,1]
      outputs += [probs.item()]
    
    return pd.DataFrame( {'score':outputs} )

# COMMAND ----------

# MAGIC %md With our wrapper defined, we now can retrieve the PyTorch model we persisted in our last notebook:

# COMMAND ----------

# DBTITLE 1,Retrieve Original Model
from mlflow.tracking import MlflowClient
client = MlflowClient()

# retrieve last run for this model
last_run = client.search_runs(
  [config['experiment_id']],  # config variable set in notebook 01
  filter_string="tags.`mlflow.runName`='{0}'".format(config['tuned_model_name_73']), 
  order_by=['start_time DESC'],
  max_results=1
  )

# retrieve model from this last run
if last_run is not None: 
  model = mlflow.pytorch.load_model('runs:/{0}/model'.format(last_run[0].info.run_id))
else:
  raise Exception('The run named "{0}" was not found.  Please make sure you have run the prior notebooks in this series.'.format(config['tuned_model_name_73']))

# COMMAND ----------

# MAGIC %md We now need to consider how to save our model with its custom wrapper.  If we intend to deploy our model to as a microservice, we should take advantage of the [mlflow model registry](https://www.mlflow.org/docs/latest/model-registry.html).  If we intend to deploy to an edge device, we may wish to leverage basic mlflow model persistence to avoid the addition of dependencies we may not be able to support on the edge device.  And if we intend to deploy to an ETL function, we can take either path:

# COMMAND ----------

# DBTITLE 1,Determine Whether to Save to Registry
# identify numerical version of databricks runtime
dbx_version = spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion").split("-")[0].split(".")

# save to registry if runtime is 8 or higher
if int(dbx_version[0]) < 8:
  save_to_registry = False
else:
  save_to_registry = True
  
print('Save to Registry?: {0}'.format(save_to_registry))

# COMMAND ----------

# MAGIC %md For the Databricks 7.3 ML LTS runtime, our persistence pattern is pretty straightforward:

# COMMAND ----------

# DBTITLE 1,Persist Model Outside of Registry (Databricks 7.3 ML LTS)
if not save_to_registry:
  
  # define model dependencies
  env = mlflow.pytorch.get_default_conda_env()
#   env['dependencies'][-1]['pip'] = [f'torch=={torch.__version__}' if d.startswith("torch==") else d for d in env['dependencies'][-1]['pip'] ] # make sure the torch version contains a local version label (+cpu)
  env['dependencies'][-1]['pip'] += [
      f'torchvision=={torchvision.__version__.split("+")[0]}',
      f'pillow=={PIL.__version__}'
    ]

  # wrap model
  wrapped_model = CVModelWrapper(model)

  # persist model to mlflow
  with mlflow.start_run(run_name=config['final_model_name']) as run:

    mlflow.pyfunc.log_model(
      artifact_path='model',
      python_model=wrapped_model,
      conda_env=env
      )

    model_name = 'runs:/{0}/model'.format(run.info.run_id)

# COMMAND ----------

# MAGIC %md We are now ready to employ our model against images.

# COMMAND ----------

# MAGIC %md ## Step 2: ETL Deployment
# MAGIC 
# MAGIC Registered models may be deployed in a number of ways.  If our goal is to score images as they are received, we might retrieve our model to a Spark user-defined function:
# MAGIC 
# MAGIC **NOTE** This deployment path works with either of the runtimes used with this and the **04a** notebook.

# COMMAND ----------

# DBTITLE 1,Retrieve Model to Spark UDF
cv_func = (
  mlflow.pyfunc.spark_udf(
    spark, 
    model_name, # this was determined above based on our runtime version
    result_type="double"
    )
  )

print('Function instantiated from model {0}'.format(model_name))

# COMMAND ----------

# MAGIC %md We could then use this function in either a batch or streaming ETL cycle to score our images.  Here we'll re-read our incoming image files to simulate a stream of new images.  Notice in the *withColumn* method call that employs our model-function, we are wrapping the *content* column within a struct data type.  This allows the column name *content* which is expected by the logic within our wrapper to be passed in with the incoming pandas dataframe:

# COMMAND ----------

# DBTITLE 1,Score Streaming Image Data
# set processing limit for reading files
max_bytes_per_executor = 512 * 1024**2 # 512-MB limit

(spark
  .readStream
  .format('cloudFiles')  # auto loader
  .option('cloudFiles.format', 'binaryFile') # read as binary image
  .option('recursiveFileLookup', 'true')
  .option('cloudFiles.includeExistingFiles', 'true') 
  .option('pathGlobFilter', '*.jpg') 
  .option('cloudFiles.maxBytesPerTrigger', sc.defaultParallelism * max_bytes_per_executor) 
  .load(config['incoming_image_file_path']) # location to read from
  .withColumn('score', cv_func(f.struct(f.col('content'))))  # score images
  .select('path','score')
  .writeStream
  .trigger(once=True) # feel free to choose any other trigger type for continuous processing
  .format("delta")
  .option("checkpointLocation", config["checkpoint_path_inference_73"])
  .table(config['scored_images_73_table'])
  )

# COMMAND ----------

# DBTITLE 1,Display Inference Results
display(spark.table(config['scored_images_73_table']))

# COMMAND ----------

# MAGIC %md ## Step 2: Edge Device Deployment (Databricks 7.3 ML LTS)
# MAGIC 
# MAGIC Our Raspberry Pi 4 device, to which we wish to deploy our model, runs the [Raspbian Buster Lite (64-bit)](https://downloads.raspberrypi.org/raspios_lite_armhf_latest) operating system which includes support for Python 3.7. We have enabled [SSH](https://linuxize.com/post/how-to-enable-ssh-on-raspberry-pi/) and [WiFi](https://raspberrytips.com/raspberry-pi-wifi-setup/) so that we may access it over the network.
# MAGIC 
# MAGIC **NOTE** Different devices will have different paths to deployment.  We've chosen to work with our original Raspberry Pi device as it presents several common challenges associated with deployment to an ARM processor. We've also simplified the deployment path to highlight a lowest-common-denominator approach that may be more widely accessible across devices.  More sophisticated deployments using Docker and/or Flask may be more appropriate depending on individual circumstances.
# MAGIC 
# MAGIC With the OS installed and configured, our next action is to verify the OS is the 64-bit version and the ARM processor is recognized appropriately.  The bitness and processor architecture determine which versions of the PyTorch and TorchVision libraries we will need to download later:

# COMMAND ----------

# DBTITLE 1,Verify CPU Architecture
# MAGIC %md **TO BE RUN ON THE RASPBERRY PI DEVICE**
# MAGIC ```
# MAGIC # verify 64 bit version of operating system
# MAGIC arch=$(lscpu | grep Architecture: | cut -d ':' -f 2 | sed -e 's/^[[:space:]]*//')
# MAGIC version=$(echo $arch | cut -c1-5)
# MAGIC if [ "$version" != "armv7" ]; then
# MAGIC   clear
# MAGIC   echo "Unexpected CPU architecture detected"
# MAGIC else
# MAGIC   clear
# MAGIC   echo "CPU architecture verified"
# MAGIC fi
# MAGIC 
# MAGIC ```

# COMMAND ----------

# MAGIC %md Assuming we see the *CPU architecture verified* message, we now can install the various packages, libraries and dependencies for our model deployment:

# COMMAND ----------

# DBTITLE 1,Install Required Libraries & Packages
# MAGIC %md **TO BE RUN ON THE RASPBERRY PI DEVICE**
# MAGIC ```
# MAGIC yes | sudo apt-get update
# MAGIC yes | sudo apt-get upgrade
# MAGIC yes | sudo apt-get install unzip
# MAGIC yes | sudo apt-get install wget
# MAGIC 
# MAGIC yes | sudo apt-get install python3
# MAGIC sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 1
# MAGIC 
# MAGIC yes | sudo apt-get install python3-pip
# MAGIC sudo update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
# MAGIC 
# MAGIC yes | sudo apt-get install ninja-build git cmake
# MAGIC 
# MAGIC yes | sudo apt-get install libopenmpi-dev libomp-dev ccache
# MAGIC yes | sudo apt-get install libopenblas-dev libblas-dev libeigen3-dev libopenblas-base libatlas-base-dev
# MAGIC 
# MAGIC yes | sudo -H pip install -U --user wheel mock pillow
# MAGIC yes | sudo -H pip install -U setuptools
# MAGIC yes | sudo -H pip install -U mlflow
# MAGIC yes | sudo -H pip install -U cloudpickle
# MAGIC 
# MAGIC yes | sudo -H pip install python3-picamera
# MAGIC yes | sudo -H pip install "picamera[array]"
# MAGIC 
# MAGIC ```

# COMMAND ----------

# MAGIC %md On a fresh operating system install, we need to run *raspi-config* to enable the picamera.  From the Configuration Tool display, select *3 Interface Options* followed by *P1 Camera* and then *Yes* at the prompt:

# COMMAND ----------

# MAGIC %md **TO BE RUN ON THE RASPBERRY PI DEVICE**
# MAGIC ```
# MAGIC sudo raspi-config
# MAGIC ```

# COMMAND ----------

# MAGIC %md Now we can install PyTorch and TorchVision.  It's important to note that these libraries are not officially supported on ARM architectures so its possible this code may not work as expected over time:

# COMMAND ----------

# DBTITLE 1,Install PyTorch
# MAGIC %md **TO BE RUN ON THE RASPBERRY PI DEVICE**
# MAGIC ```
# MAGIC # setup download directory
# MAGIC dir="$(pwd)/download"
# MAGIC if [[ ! -e $dir ]]; then
# MAGIC     mkdir $dir
# MAGIC fi
# MAGIC 
# MAGIC # download whl and checksum files
# MAGIC TORCHFILE="torch-1.7.0a0-cp37-cp37m-linux_armv7l.whl"
# MAGIC TORCHCHK="$TORCHFILE.sha256"
# MAGIC if [ -f "$dir/$TORCHFILE" ]; then
# MAGIC     rm $dir/$TORCHFILE
# MAGIC fi 
# MAGIC wget https://github.com/ljk53/pytorch-rpi/raw/master/$TORCHFILE -P $dir
# MAGIC wget https://github.com/ljk53/pytorch-rpi/raw/master/$TORCHCHK -P $dir
# MAGIC 
# MAGIC # verify checksum
# MAGIC cd $dir
# MAGIC echo "$(cat $TORCHCHK)" | sha256sum --check
# MAGIC cd ~
# MAGIC 
# MAGIC # install whl
# MAGIC pip install "$dir/$TORCHFILE"
# MAGIC 
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Install Torchvision
# MAGIC %md **TO BE RUN ON THE RASPBERRY PI DEVICE**
# MAGIC ```
# MAGIC # setup download directory
# MAGIC dir="$(pwd)/download"
# MAGIC if [[ ! -e $dir ]]; then
# MAGIC     mkdir $dir
# MAGIC fi
# MAGIC 
# MAGIC # download whl
# MAGIC VISIONFILE="torchvision-0.4.0a0+d31eafa-cp37-cp37m-linux_armv7l.whl"
# MAGIC VISIONFILEURL="https://github.com/nmilosev/pytorch-arm-builds/raw/master/torchvision-0.4.0a0%2Bd31eafa-cp37-cp37m-linux_armv7l.whl"
# MAGIC if [ -f "$dir/$VISIONFILE" ]; then
# MAGIC     rm $dir/$VISIONFILE
# MAGIC fi 
# MAGIC wget $VISIONFILEURL -P $dir
# MAGIC 
# MAGIC # install whl
# MAGIC yes | sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev
# MAGIC yes | sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev
# MAGIC pip install "$dir/$VISIONFILE"
# MAGIC 
# MAGIC ```

# COMMAND ----------

# MAGIC %md With our dependencies in place, we can now retrieve our model from our Databricks mlflow deployment.  To do this, we need to update some environment variables so that the local mlflow components understand how to connect to our remote workspace.
# MAGIC 
# MAGIC There are three variables we need to set: MLFLOW_TRACKING_URI, DATABRICKS_HOST & DATABRICKS_TOKEN.  The MFLOW_TRACKING_URI variable is set to *databricks* to indicate that we are connecting to a Databricks-integrated deployment of mlflow.  The DATABRICKS_HOST variable is set to the HTTPS address of our Databricks workspace.  And the DATABRICKS_TOKEN variable is set to the value of a [personal access token](https://docs.databricks.com/dev-tools/api/latest/authentication.html) with access to the environment.  You might wish to persist these values in the local */etc/environment* file so that they might be retained between device shutdowns but for demonstration purposes we'll perform a simple export:

# COMMAND ----------

# DBTITLE 1,Configure Access to Remote Mlflow
# MAGIC %md **TO BE RUN ON THE RASPBERRY PI DEVICE**
# MAGIC ```
# MAGIC export MLFLOW_TRACKING_URI=databricks
# MAGIC export DATABRICKS_HOST=<YOUR WORKSPACE URL HERE>
# MAGIC export DATABRICKS_TOKEN=<YOUR PERSONAL ACCESS TOKEN HERE>
# MAGIC ```

# COMMAND ----------

# MAGIC %md We are now going to download the artifacts for the last model we persisted, under the assumption this is the model version we want.  If this isn't the right logic, we might explore using [tags](https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_tag) or alternative names for models persisted outside the registry to ensure we find the right model: 

# COMMAND ----------

# DBTITLE 1,Download Mlflow Model to Local Device
# MAGIC %md **TO BE RUN ON THE RASPBERRY PI DEVICE**
# MAGIC ```
# MAGIC # name of model 
# MAGIC final_model_name="cv pytorch final"
# MAGIC 
# MAGIC # create clean directory
# MAGIC dir="$(pwd)/models"
# MAGIC if [ -d "$dir" ]; then
# MAGIC     rm -r $dir
# MAGIC fi
# MAGIC mkdir $dir
# MAGIC 
# MAGIC # write model artifact retrieval script
# MAGIC download_script="download_model_artifacts.py"
# MAGIC 
# MAGIC echo "import mlflow" > $download_script
# MAGIC echo "client = mlflow.tracking.MlflowClient()" >> $download_script
# MAGIC echo "" >> $download_script
# MAGIC echo "last_run = client.search_runs(" >> $download_script
# MAGIC echo "   [e.experiment_id for e in client.list_experiments()]," >> $download_script
# MAGIC echo "   filter_string=\"tags.\`mlflow.runName\`='{0}'\".format('$final_model_name')," >> $download_script
# MAGIC echo "   order_by=['start_time DESC']," >> $download_script
# MAGIC echo "   max_results=1" >> $download_script
# MAGIC echo "   )[0]" >> $download_script
# MAGIC echo "" >> $download_script
# MAGIC echo "client.download_artifacts(last_run.info.run_id, 'model', \"$dir\")" >> $download_script
# MAGIC 
# MAGIC # execute script
# MAGIC python $download_script
# MAGIC ```

# COMMAND ----------

# MAGIC %md And now on our Raspberry Pi, we can use the model to score an image:

# COMMAND ----------

# DBTITLE 1,Score a New Image with Trained Model
# MAGIC %md **TO BE RUN ON THE RASPBERRY PI DEVICE**
# MAGIC ```
# MAGIC # write image capture and scoring script
# MAGIC scoring_script="image_capture_scoring.py"
# MAGIC 
# MAGIC echo "from picamera import PiCamera" > $scoring_script
# MAGIC echo "import io, os" >> $scoring_script
# MAGIC echo "import pandas as pd" >> $scoring_script
# MAGIC echo "import mlflow" >> $scoring_script
# MAGIC echo "" >> $scoring_script
# MAGIC echo "# output file" >> $scoring_script
# MAGIC echo "local_filename = '/home/pi/images/image.jpg'" >> $scoring_script
# MAGIC echo "model_folder = '/home/pi/models/model'" >> $scoring_script
# MAGIC echo "" >> $scoring_script
# MAGIC echo "# clean up image capture dir" >> $scoring_script
# MAGIC echo "path = '/'.join(local_filename.split('/')[:-1])" >> $scoring_script
# MAGIC echo "if not os.path.exists(path):" >> $scoring_script
# MAGIC echo "  os.makedirs(path)" >> $scoring_script
# MAGIC echo "" >> $scoring_script
# MAGIC echo "# delete image file if exists" >> $scoring_script
# MAGIC echo "if os.path.exists(local_filename):" >> $scoring_script
# MAGIC echo "    os.remove(local_filename)" >> $scoring_script
# MAGIC echo "" >> $scoring_script
# MAGIC echo "# capture image" >> $scoring_script
# MAGIC echo "camera = PiCamera()" >> $scoring_script
# MAGIC echo "camera.resolution = (600, 600)" >> $scoring_script
# MAGIC echo "camera.framerate = 15" >> $scoring_script
# MAGIC echo "camera.start_preview()" >> $scoring_script
# MAGIC echo "camera.capture(local_filename)" >> $scoring_script
# MAGIC echo "camera.stop_preview()" >> $scoring_script
# MAGIC echo "" >> $scoring_script
# MAGIC echo "# read image to pandas dataframe" >> $scoring_script
# MAGIC echo "with open(local_filename, 'rb') as f:" >> $scoring_script
# MAGIC echo "    image_bytes = bytearray(f.read())" >> $scoring_script
# MAGIC echo "" >> $scoring_script
# MAGIC echo "image_pd = pd.DataFrame([[image_bytes]], columns=['content'])" >> $scoring_script
# MAGIC echo "" >> $scoring_script
# MAGIC echo "# score dataframe" >> $scoring_script
# MAGIC echo "model = mlflow.pyfunc.load_model(model_folder)" >> $scoring_script
# MAGIC echo "score = model.predict(image_pd)" >> $scoring_script
# MAGIC echo "score = score.iloc[0,].values[0]" >> $scoring_script
# MAGIC echo "" >> $scoring_script
# MAGIC echo "print(score)" >> $scoring_script
# MAGIC 
# MAGIC # execute script
# MAGIC python $scoring_script
# MAGIC ```

# COMMAND ----------

# MAGIC %md The code above is not intended to demonstrate a production-quality deployment to an edge device.  Instead, it's focused on the core mechanics of using the model locally in a manner that provides maximum flexibility.  Other deployment paths including the retrieval of Docker images from the mlflow repository may provide better suited for some scenarios.

# COMMAND ----------

# MAGIC %md © 2021 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
