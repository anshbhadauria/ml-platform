import os
import sys
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
import click
import dvc.api
import logging
import git
import ast

sys.path.append(os.getcwd())

from infra import utils

config = utils.get_configuration("config.yml")
logging_config= utils.get_configuration("logging_config.yml")
logging.config.dictConfig(logging_config)
logger= logging.getLogger(__name__)

mlflow_artifact_uri = os.path.join(
    "s3://",
    config.s3_bucketname,
    config.project_name
    )
os.environ["MLFLOW_S3_ENDPOINT_URL"] = config.s3_endpointurl
os.environ["MLFLOW_TRACKING_URI"] = config.mlflow_tracking_uri
experiment_name = config.project_name

try:
    experiment_id=mlflow.create_experiment(
        name=experiment_name,
        artifact_location=mlflow_artifact_uri
    )
except:
    mlflow.set_experiment(experiment_name=experiment_name)
    experiment_id=mlflow.get_experiment_by_name(experiment_name).experiment_id

def mlflow_run(i, o, params):

    with mlflow.start_run(experiment_id=experiment_id):
        
        df = utils.get_url(i, sep=";")
        logger.info("Training columns: %s", str(df.columns))
        train_df = df.loc[:, df.columns != 'quality']
        clf=RandomForestClassifier(**params)
        logger.info("Training parameters: %s", params)
        clf.fit(train_df, df.quality)
        signature = infer_signature(
            train_df, 
            clf.predict(train_df)
            )
        mlflow.log_params(clf.get_params())
        mlflow.set_tags({"dataset":i})
        mlflow.sklearn.log_model(
            clf, 
            o, 
            signature = signature,
            registered_model_name = o,    
            )

        return

@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option("--i",
default=None,
help="Path to training dataset."
)
@click.option("--o",
default=None,
help="Name of trained model."
)
@click.option("--p",
default=None,
help="Parameters for training model."
)
def train(i, o, p=None):
    
    if p is not None:
        for hp in ast.literal_eval(p):    
            mlflow_run(i, o, hp)
    else:
        mlflow_run(i, o, {"max_depth":7, "random_state":0})
    return

if __name__=="__main__":
    train()


