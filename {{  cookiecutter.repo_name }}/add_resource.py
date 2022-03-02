import click
import dvc.api
from infra import utils
import subprocess
import logging
from box import Box
import yaml
from collections import defaultdict
import yaml
from datetime import datetime
import git

repo = git.Repo(search_parent_directories=True)
sha = repo.head.commit.hexsha

now = datetime.now()

config=utils.get_configuration("config.yml")

logging_config= utils.get_configuration("logging_config.yml")
logging.config.dictConfig(logging_config)
logger= logging.getLogger(__name__)

def get_resource_url(resource):

    return dvc.api.get_url(
    resource,
    repo=".git/"
    )

def prep_string(resource, version):
    s = {resource: 
            {
                "url": get_resource_url(resource),
                "version": version,
                "timestamp": now.strftime("%m/%d/%Y,%H:%M:%S"),
                "commit": sha
            }
        }
    return yaml.dump(s)

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

    dvc_rep=utils.get_configuration(
        "dvc_resources.yml"
        )

    commands = [
    f"dvc add {resource}",
    "dvc push",
    f"git add {resource}.dvc",
    f"git commit -m \"{resource}.dvc\""

    ]   

    with open("log.txt", "a") as logs:
        for c in commands:
            logger.info("Running cmd %s", c)
            subprocess.run(c.split(" "), stderr=logs)
    
    logger.info("List of files tracked by dvc")

    show_dvc_tracked_files()

    if resource in dvc_rep:
        if get_resource_url(resource)==dvc_rep[resource]["url"]:
            logger.info(f"Resource {resource} is already tracked by dvc.")
        else:
            w=prep_string(resource, dvc_rep[resource]["version"]+1)    
            with open("conf/dvc_resources.yml", 'a') as yaml_file:
                yaml_file.write(w)
            logger.info(f"Created a new version of {resource}.")
    elif resource not in dvc_rep:
        w=prep_string(resource, 1)
        with open("conf/dvc_resources.yml", 'a') as yaml_file:
            yaml_file.write(w)
        logger.info(f"Resource {resource} is now being tracked by dvc.")
    
    else:
        logger.info(f"Resource {resource} is already tracked by dvc.")

    return 

if __name__=="__main__":
    add_resource_to_dvc()
