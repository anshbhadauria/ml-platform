import click
import dvc.api
from infra import utils
import subprocess
import logging
from box import Box
import yaml
from collections import defaultdict

config=utils.get_configuration("config.yml")

logging_config= utils.get_configuration("logging_config.yml")
logging.config.dictConfig(logging_config)
logger= logging.getLogger(__name__)

def get_resource_url(resource):

    return dvc.api.get_url(
    resource,
    repo=".git/"
    )

def show_dvc_tracked_files():
    commands = [
        "dvc list . --recursive --dvc-only"
    ]   
    with open("log.txt", "a") as logs:
        for c in commands:
            logger.info("Running cmd %s", c)
            subprocess.run(c.split(" "), stderr=logs)
    

@click.command()
@click.option("--resource",
default=None,
help="Path to training dataset."
)
def add_resource_to_dvc(resource):

    dvc_rep=utils.get_configuration("dvc_resources.yml")

    if resource not in dvc_rep:
        commands = [
        f"dvc add {resource}",
        "dvc push"
        ]   
        with open("log.txt", "a") as logs:
            for c in commands:
                logger.info("Running cmd %s", c)
                subprocess.run(c.split(" "), stderr=logs)
        
        logger.info("List of files tracked by dvc")
        show_dvc_tracked_files()

        with open("conf/dvc_resources.yml", 'a') as yaml_file:
            yaml_file.write(f"{resource}: {get_resource_url(resource)}\n")
        
        logger.info(f"Resource {resource} is now being tracked by dvc.")
    else:
        logger.info(f"Resource {resource} is already tracked by dvc.")

    return 

if __name__=="__main__":
    add_resource_to_dvc()
