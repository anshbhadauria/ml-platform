
import subprocess
from infra import utils
import logging
import sys
import os

config=utils.get_configuration("config.yml")

logging_config= utils.get_configuration("logging_config.yml")
logging.config.dictConfig(logging_config)
logger= logging.getLogger(__name__)

commands=[
"git init",
"git add .",
"git commit -m '"'init'"' ",
"dvc init",
"dvc remote add -d " + config.project_name +" s3://"+ config.s3_bucketname +"/"+config.project_name,
"dvc remote modify " + config.project_name + " endpointurl " + config.s3_endpointurl,
"dvc config core.autostage true"
]

with open("log.txt","w") as logs:
    for c in commands:
        logger.info("Running cmd %s", c)
        subprocess.run(c.split(" "), stdout=logs, stderr=logs)
