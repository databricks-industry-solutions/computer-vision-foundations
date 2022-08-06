# Databricks notebook source
# MAGIC %md The purpose of this notebook is to demonstrate patterns for computer vision model deployment.  This notebook follows notebook **03a**.  

# COMMAND ----------

# MAGIC %md ## Introduction
# MAGIC 
# MAGIC In notebook 04a and 04b, we explore three different deployment vehicles:</p>
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/cv_deploy4.png' width=550>
# MAGIC 
# MAGIC Each deployment path is facilitated by mlflow, a model management technology integrated with the Databricks workspace.  For the ETL function and microservice deployment paths, it is recommended you use a recent version of Databricks runtimes.  

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
import time


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
  filter_string="tags.`mlflow.runName`='{0}'".format(config['tuned_model_name']), 
  order_by=['start_time DESC'],
  max_results=1
  )

# retrieve model from this last run
if last_run is not None: 
  model = mlflow.pytorch.load_model('runs:/{0}/model'.format(last_run[0].info.run_id))
else:
  raise Exception('The run named "{0}" was not found.  Please make sure you have run the prior notebooks in this series.'.format(config['tuned_model_name']))

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

# MAGIC %md For the model registry, we can more easily discover our preferred version of a persisted model but this requires us to do a bit more model management.
# MAGIC 
# MAGIC First, we'll make sure we have no other *production* instances of this model in the registry:

# COMMAND ----------

# DBTITLE 1,Archive Any Prior Model Registrations
if save_to_registry:

  try:
    for m in client.get_latest_versions(config['final_model_name'], stages=['Production']):
      client.transition_model_version_stage(
        name=m.name,
        version=m.version,
        stage='Archived'
        )
  except:
    pass

# COMMAND ----------

# MAGIC %md And now we persist our model to the registry:

# COMMAND ----------

# DBTITLE 1,Persist Model to Registry
if save_to_registry:
  # define model dependencies
  env = mlflow.pytorch.get_default_conda_env()
  env['dependencies'][-1]['pip'] += [
      f'torchvision=={torchvision.__version__.split("+")[0]}',
      f'pillow=={PIL.__version__}'
    ]

  # wrap model
  wrapped_model = CVModelWrapper(model)

  # persist model to registry
  with mlflow.start_run(run_name=config['final_model_name']) as run:

    mlflow.pyfunc.log_model(
      artifact_path='model',
      python_model=wrapped_model,
      registered_model_name=config['final_model_name'],
      conda_env=env
      )

# COMMAND ----------

# MAGIC %md The mlflow registry allows us to manage models as part of an MLOps pipeline.  Most models would undergo testing before being elevated to production, and this elevation would often take place through automation or the built-in UI.  But we are going to skip that and programmatically elevate our just registered model to production now:

# COMMAND ----------

# DBTITLE 1,Promote Model to Production
if save_to_registry:
  
  for m in client.get_latest_versions(config['final_model_name'], stages=['None']):
    client.transition_model_version_stage(
      name=m.name,
      version=m.version,
      stage='Production'
      )
    
    model_name = 'models:/{0}/Production'.format(config['final_model_name'])

# COMMAND ----------

# MAGIC %md We are now ready to employ our model against images.

# COMMAND ----------

# MAGIC %md ## Step 2: ETL Deployment
# MAGIC 
# MAGIC Registered models may be deployed in a number of ways.  If our goal is to score images as they are received, we might retrieve our model to a Spark user-defined function:
# MAGIC 
# MAGIC **NOTE** This deployment path works with either of the runtimes used with this notebook.

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

# define stream
(
  spark
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
  .option("checkpointLocation", config["checkpoint_path_inference"])
  .table(config['scored_images_table'])
  )

# COMMAND ----------

display(spark.table(config['scored_images_table']))

# COMMAND ----------

# MAGIC %md ## Step 3: Databricks Model Serving Deployment
# MAGIC 
# MAGIC If our needs required us to deploy a centralized service that any number of applications could call on-demand, we might consider a different deployment path for our model.  Instead of deploying the model to a function, we might deploy it to a Docker image and expose it through a REST API.  Integration between mlflow and [Azure ML](https://www.mlflow.org/docs/latest/python_api/mlflow.azureml.html) and [AWS Sagemaker](https://www.mlflow.org/docs/latest/python_api/mlflow.sagemaker.html) make this a relatively simple process.  We might also take advantage of Databricks's [model serving](https://docs.databricks.com/applications/mlflow/model-serving.html?_ga=2.81483735.1331949122.1632064157-31266672.1629894740) capabilities.
# MAGIC 
# MAGIC To leverage Databricks model serving, we will need to make use of our workspace's user-interface.  To do this, switch into the Databricks UI's **Machine Learning UI** by clicking the drop-down in the left-hand panel of the Databricks UI. Once in the Machine Learning UI, click the **Models** icon in the left-hand side of the screen.
# MAGIC 
# MAGIC In the resulting page, locate the model registered above. Click on the model and note that two tabs are available: Details & Serving. Clicking the **Serving** tab, we can select the **Enable Serving** button to launch a small, single-node cluster to host the model behind an autogenerated REST API. Wait for the production version of the model to appear in the Model Versions table with a status of *Ready*.  (Once you are done with the REST API, please be sure to return to this interface to stop the hosting cluster.)
# MAGIC 
# MAGIC Once the model is in a Ready state, copy the model URL that identifies the production version of your model and update the cell below with that string:

# COMMAND ----------

# DBTITLE 1,Get Model URL
workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None) 
model_url = f"{workspace_url}/model/{config['final_model_name']}/Production/invocations"

# COMMAND ----------

# MAGIC %md Finally, the REST API presented by Databricks Model Serving is secured using a [Databricks Personal Access Token](https://docs.databricks.com/dev-tools/api/latest/authentication.html).

# COMMAND ----------

# DBTITLE 1,Get Personal Access Token
personal_access_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

# MAGIC %md Let's now consider the data we will pass to the REST API.  To get started, let's retrieve a few images we wish to score to a pandas dataframe:

# COMMAND ----------

# DBTITLE 1,Retrieve Sample Images
images = (
  spark
    .table(config['input_images_table'])
    .select('content')
    .sample(withReplacement=False, fraction=0.01)
    .toPandas()
  )

images

# COMMAND ----------

# MAGIC %md Notice the *content* field contains byte array values for the images.  To pass these over the network to the REST API, we need to base-64 encode the values and then convert those values to a UTF-8 string as is standard for the transfer of binary payloads over HTTP:

# COMMAND ----------

# DBTITLE 1,Base-64 Encode Images & Convert to UTF-8
images['content'] = (
  images['content'].apply(
    lambda b: base64.b64encode(b).decode('utf-8')
    )
  )

images

# COMMAND ----------

# MAGIC %md Now we can export the pandas dataframe to a JSON format that can be sent to the API:

# COMMAND ----------

# DBTITLE 1,Export Pandas DataFrame to JSON
data_json = images.to_dict(orient='records')
data_json

# COMMAND ----------

# MAGIC %md Now, let's send this data to the REST API:

# COMMAND ----------

model_url

# COMMAND ----------

# DBTITLE 1,Call REST API to Score Images
import time
if save_to_registry:
  time.sleep(600) # wait for new version to finish deployment

  # send data to REST API for scoring
  headers = {'Authorization': 'Bearer {0}'.format(personal_access_token)}
  response = requests.request(method='POST', headers=headers, url=model_url, json=data_json)
  
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')

  # display returned scores
  print(response.json())

# COMMAND ----------

# MAGIC %md The code above is not intended to demonstrate a production-quality deployment to an edge device.  Instead, it's focused on the core mechanics of using the model locally in a manner that provides maximum flexibility.  Other deployment paths including the retrieval of Docker images from the mlflow repository may provide better suited for some scenarios.

# COMMAND ----------

# MAGIC %md © 2021 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
