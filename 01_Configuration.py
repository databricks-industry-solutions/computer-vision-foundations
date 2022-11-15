# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/computer-vision-foundations. For more information about this solution accelerator, visit https://www.databricks.com/blog/2021/12/17/enabling-computer-vision-applications-with-the-data-lakehouse.html.

# COMMAND ----------

# MAGIC %md The purpose of this notebook is to manage configuration settings for the notebooks in the computer vision solution accelerator. Please adapt these settings as needed to align the notebook deployment with your environment. 

# COMMAND ----------

# DBTITLE 1,Initialize Configuration Variable
if 'config' not in locals():
  config = {}

# COMMAND ----------

# DBTITLE 1,Storage Settings
config['mount_point'] = '/tmp/cv_foundations/' # location for data files; use a mount path like /mnt/images/ if reading from external storage
config['database_root'] = '/tmp/cv_foundations/cv/'
config['raw_image_file_path'] = "s3://db-gtm-industry-solutions/data/rcg/cv/images/"  # where incoming image files land - this is a publically accessible S3 bucket
config['incoming_image_file_path'] = config['mount_point'] + 'tmp/incoming_image_file_path/'
config['checkpoint_path'] = config['mount_point'] + 'tmp/image_processing_chkpnt/' # folder where incoming image processing checkpoint resides
config['checkpoint_path_inference'] = config['mount_point'] + 'tmp/image_inference_chkpnt/'
config['checkpoint_path_inference_73'] = config['mount_point'] + 'tmp/image_inference_chkpnt_73/'
config['petastorm_path'] = 'file:///dbfs/tmp/petastorm/cache' # location where to store petastorm cache files
config['input_images_table'] = 'cv.images'
config['scored_images_73_table'] = "cv.scored_images_73"
config['scored_images_table'] = "cv.scored_images"

# COMMAND ----------

# DBTITLE 1,Model Settings
config['tuning_model_name'] = 'cv pytorch tuning'
config['tuned_model_name'] = 'cv pytorch tuned'
config['tuned_model_name_73'] = 'cv pytorch tuned 73'
config['final_model_name'] = 'cv pytorch final'

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
useremail = spark.sql('select current_user() as user').collect()[0]['user']
experiment_name = f"/Users/{useremail}/computer_vision_foundations"
mlflow.set_experiment(experiment_name) 
config['experiment_id'] = (
    MlflowClient()
    .get_experiment_by_name(experiment_name) 
    .experiment_id
)

# COMMAND ----------

# MAGIC %md © 2021 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
