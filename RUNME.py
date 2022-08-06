# Databricks notebook source
# MAGIC %md This notebook sets up the companion cluster(s) to run the solution accelerator. It also creates the Workflow to create a Workflow DAG and illustrate the order of execution. Feel free to interactively run notebooks with the cluster or to run the Workflow to see how this solution accelerator executes. Happy exploring!
# MAGIC 
# MAGIC The pipelines, workflows and clusters created in this script are not user-specific, so if another user alters the workflow and cluster via the UI, running this script again after the modification resets them.
# MAGIC 
# MAGIC **Note**: If the job execution fails, please confirm that you have set up other environment dependencies as specified in the accelerator notebooks. Accelerators sometimes require the user to set up additional cloud infra or data access, for instance. 

# COMMAND ----------

# DBTITLE 0,Install util packages
# MAGIC %pip install git+https://github.com/databricks-academy/dbacademy-rest git+https://github.com/databricks-academy/dbacademy-gems git+https://github.com/databricks-industry-solutions/notebook-solution-companion

# COMMAND ----------

from solacc.companion import NotebookSolutionCompanion

# COMMAND ----------

job_json = {
        "timeout_seconds": 14400,
        "max_concurrent_runs": 1,
        "tags": {
            "usage": "solacc_testing",
            "group": "RCG"
        },
        "tasks": [
            
            {
                "job_cluster_key": "cvf_cluster",
                "notebook_task": {
                    "notebook_path": f"01_Configuration"
                },
                "task_key": "cvf_01"
            },
            {
                "job_cluster_key": "cvf_cluster",
                "notebook_task": {
                    "notebook_path": f"02_Data Ingest"
                },
                "task_key": "cvf_02",
                "depends_on": [
                    {
                        "task_key": "cvf_01"
                    }
                ]
            },
            {
                "job_cluster_key": "cvf_cluster",
                "notebook_task": {
                    "notebook_path": f"03a_Model Training"
                },
                "task_key": "cvf_03",
                "depends_on": [
                    {
                        "task_key": "cvf_02"
                    }
                ]
            },
            {
                "job_cluster_key": "cvf_cluster",
                "notebook_task": {
                    "notebook_path": f"04a_Model Deployment"
                },
                "task_key": "cvf_04",
                "depends_on": [
                    {
                        "task_key": "cvf_03"
                    }
                ]
            },
            {
                "job_cluster_key": "cvf_cluster_DBR73",
                "notebook_task": {
                    "notebook_path": f"03b_Model Training"
                },
                "task_key": "cvf_05",
                "depends_on": [
                    {
                        "task_key": "cvf_04"
                    }
                ]
            },
            {
                "job_cluster_key": "cvf_cluster_DBR73",
                "notebook_task": {
                    "notebook_path": f"04b_Model Deployment"
                },
                "task_key": "cvf_06",
                "depends_on": [
                    {
                        "task_key": "cvf_05"
                    }
                ]
            }
        ],
        "job_clusters": [
            {
                "job_cluster_key": "cvf_cluster",
                "new_cluster": {
                    "spark_version": "10.4.x-cpu-ml-scala2.12",
                "spark_conf": {
                    "spark.databricks.delta.formatCheck.enabled": "false"
                    },
                    "num_workers": 2,
                    "node_type_id": {"AWS": "i3.xlarge", "MSA": "Standard_D3_v2", "GCP": "n1-highmem-4"},
                    "custom_tags": {
                        "usage": "solacc_testing"
                    },
                }
            },
            {
                "job_cluster_key": "cvf_cluster_DBR73",
                "new_cluster": {
                    "spark_version": "7.3.x-cpu-ml-scala2.12",
                "spark_conf": {
                    "spark.databricks.delta.formatCheck.enabled": "false"
                    },
                    "num_workers": 2,
                    "node_type_id": {"AWS": "i3.xlarge", "MSA": "Standard_D3_v2", "GCP": "n1-highmem-4"},
                    "custom_tags": {
                        "usage": "solacc_testing"
                    },
                }
            }
        ]
    }

# COMMAND ----------

dbutils.widgets.dropdown("run_job", "False", ["True", "False"])
run_job = dbutils.widgets.get("run_job") == "True"
NotebookSolutionCompanion().deploy_compute(job_json, run_job=run_job)

# COMMAND ----------


