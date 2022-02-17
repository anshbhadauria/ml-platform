
import yaml
import sqlalchemy
from sqlalchemy.engine.url import URL
from sqlalchemy import Table
import pandas as pd
import logging
import logging.config
import sys
import joblib
import os
import csv
from io import StringIO
from psycopg2 import sql
import datetime
import psycopg2
import pprint
from contextlib import contextmanager
import multiprocessing as mp
import functools
from pandas.api.types import is_numeric_dtype, is_string_dtype
from os import listdir
from os.path import isfile, join
import re
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
from box import Box
from urllib.parse import urlparse

cur_dir = os.path.dirname(os.path.realpath(__file__))
artifacts_path = cur_dir + "/../artifacts/"

def get_configuration(config_filename):
#Get and read configuration.
    config_file = cur_dir + "/../conf/"+ config_filename
    with open(config_file) as ymlfile:
        cfg= yaml.safe_load(ymlfile)    
    return Box(cfg)

config=get_configuration("config.yml")
logging_config= get_configuration("logging_config.yml")
logging.config.dictConfig(logging_config)
logger= logging.getLogger(__name__)

##############################################################

def get_file(filename):
    try: 
        return joblib.load(artifacts_path+filename)
    except Exception as e:
        logger.error(e)


def get_file_or_none(filename):
    """
    Loads the file using joblib. Returns None if file not found.

    @param filename
    @return Joblib-loaded file, or None is not found
    """
    full_filename = artifacts_path+filename
    if os.path.isfile(full_filename):
        return get_file(filename)
    return None


def save_file(obj, filename):
    try:
        path = Path(artifacts_path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, artifacts_path+filename)    
    except Exception as e:
        print(e)
        

def get_table_name(table_type):
    return get_configuration("config.yml")["TABLES"][table_type]["NAME"]


def get_table_configuration(table_type):
    return get_configuration("config.yml")["TABLES"][table_type]


def get_engine(table_type):
    try:
        engine=sqlalchemy.create_engine(
            get_database_uri(table_type),
            echo=False
            )
        logger.info("Connected to %s", engine)
        return engine
    except Exception as e:
        logger.error(e)

def get_database_uri(table_type):

    try:
        return os.environ[get_configuration("config.yml")["TABLES"][table_type]["DATABASE"]]

    except Exception as e:
        logger.info(
            "DATABASE URI environment variable not found. Looking for credentials in config."
            )
        db = get_configuration("config.yml")["TABLES"][table_type]["DATABASE"]
        return get_configuration("config.yml")["DATABASES"][db]["URI"]


def get_database_metadata(table_type):
    try:
        return sqlalchemy.MetaData(
            get_engine(table_type)
            )
    except Exception as e:
        logger.error(e)
        

def reflected_table_object(table_type):
    try:
        return Table(
                get_table_name(table_type),
                get_database_metadata(table_type), 
                autoload_with=get_engine(table_type)
            ) 
    except Exception as e:
        logger.error(e)
###############################################################



def read_tables(tables_to_read):
    
    temp_table_list = []

    for table_type in tables_to_read:
        temp_table = read_table(table_type)
        if len(temp_table)>0:
            temp_table_list.append(temp_table)

    return pd.concat(
            temp_table_list, 
            axis=0, 
            sort=False
            )


def get_list_of_columns_to_read_for_table(table_type):
    return get_configuration("config.yml")["TABLES"][table_type]["READ_COLUMNS"]

def get_list_of_columns_to_write_for_table(table_type):
    return get_configuration("config.yml")["TABLES"][table_type]["WRITE_COLUMNS"]


def query_builder(
    table_type,
    columns_to_read=None,
    filter_column=None,
    filter_values=None,
    filter_type=None,
    limit=None
    ):
    
    if filter_column is None and filter_values is None:
        query = sql.SQL("select {fields} from {table}").format(
            fields=sql.SQL(', ').join(sql.Identifier(n) for n in columns_to_read),
            table=sql.Identifier(get_table_name(table_type))
        )
    
    elif limit is not None:
        query = sql.SQL("select {fields} from {table} where {col} "+filter_type+" %s"+" limit "+str(limit)).format(
            fields=sql.SQL(', ').join(sql.Identifier(n) for n in columns_to_read),
            table=sql.Identifier(get_table_name(table_type)),
            col=sql.Identifier(filter_column)
            
        )
        
    else:    
        query = sql.SQL("select {fields} from {table} where {col} "+filter_type+" %s").format(
            fields=sql.SQL(', ').join(sql.Identifier(n) for n in columns_to_read),
            table=sql.Identifier(get_table_name(table_type)),
            col=sql.Identifier(filter_column)
            
        )
    
    return query
            
def read_table(
    table_type, 
    columns_to_read=None,
    filter_column=None,
    filter_values=None,
    filter_type=None,
    limit=None
    ):

    """
    Read a postgresql Table using psycopg2 and return it as a pandas dataframe.
    
    """
    start_time = datetime.datetime.now()
    logger.info(
        "Beginning to read from %s data at: %s",
        table_type,
        start_time
        )
    
    db_uri = get_database_uri(table_type)
    
    conn = psycopg2.connect(db_uri)
        
    if columns_to_read is None:
        columns_to_read = [c.name for c in reflected_table_object(table_type).columns]
    
    query = query_builder(
        table_type, 
        columns_to_read=columns_to_read,
        filter_column=filter_column,
        filter_values=filter_values,
        filter_type=filter_type,
        limit=limit
    )
            
    cur = conn.cursor()
    
    if filter_column is None:
        cur.execute(query)
    else:
        cur.execute(query,(filter_values,))
        
    data = pd.DataFrame(cur.fetchall(),columns=columns_to_read)
    
    end_time = datetime.datetime.now()
    time_diff = (end_time-start_time).total_seconds()
    logger.info(
        "Read %d rows in %d minutes %d seconds", 
        data.shape[0], 
        time_diff // 60, 
        time_diff % 60
        )
    return data        


def delete_table(table_type):

    """
    Delete a SQL table.
    
    """
    
    table = reflected_table_object(table_type)
    
    if table is not None:
        logger.info(
            'Deleting %s table', 
            table_type
            )
        d = table.delete()

        d.execute()

    return


def reset_tables(list_tables_to_delete):
    
    """
    
    Delete list of table_types provided. 
    
    """
    for table_type in list_tables_to_delete:
        try:
            delete_table(table_type)
        except sqlalchemy.exc.NoSuchTableError:
            logger.info(
                "%s doesn't exist", 
                table_type
                )
    
    return


def psql_insert_copy(table, conn, keys, data_iter):
    
    """
    Utility function for converting table object to csv and then to write those rows to a postgre database.
    
    """
    dbapi_conn = conn.connection


    with dbapi_conn.cursor() as cur:
        output = StringIO()
        writer = csv.writer(output, delimiter = '\t')
        writer.writerows(data_iter)
        output.seek(0)
        cur.copy_from(
            output, 
            '"'+table.name+'"', 
            null=""
            )    
    
    return

def write_table(
    table_type, 
    data, 
    if_exists="replace", 
    index=False
    ):
    
    """
    Write a Pandas dataframe to a Postgresql table.
    
    """

    start_time = datetime.datetime.now()
    
    
    logger.info(
        "Starting writing %d rows with columns: %s at: %s",
        data.shape[0], 
        data.columns, 
        start_time
        )

    # Create table metadata
    data.head(0).to_sql(
        get_table_name(table_type), 
        get_engine(table_type), 
        if_exists=if_exists, 
        index=index
        )
    
    # Write table
    data.to_sql(
        get_table_name(table_type), 
        get_engine(table_type), 
        method=psql_insert_copy,
        if_exists=if_exists,
        chunksize=100000,
        index=False
        )

    
    end_time = datetime.datetime.now()
    time_diff = (end_time-start_time).total_seconds()
    
    logger.info(
        "Writing to %s finished in: %d minutes %d seconds", 
        get_table_name(table_type), 
        time_diff // 60, 
        time_diff % 60
        )        

    return

###############################################################
class Pipeline:

    """
    
    Create a Pipeline Object, which is basically an iterable of Python callables.

    """
    
    def __init__(self, value=None, function_pipeline=[]):
        self.value = value
        self.function_pipeline = function_pipeline
        
    def execute(self):
        return functools.reduce(lambda v, f: f(v), self.function_pipeline, self.value)
    
    def add_function(self,func):
        self.function_pipeline.append(func)
        
    def __repr__(self):
        return str(pprint.pprint({"Pipeline": [f.__name__ for f in self.function_pipeline],
                   "Dataframe shape": self.value.shape,
                   "Dataframe columns": self.value.columns}))


def fill_na(dataframe):
    
    for col in dataframe.columns:
        if is_numeric_dtype(dataframe[col]):        
            dataframe[col]=dataframe[col].fillna(0)
        elif is_string_dtype(dataframe[col]):
            dataframe[col]=dataframe[col].fillna("")
    
    return dataframe

def get_latest_model():
    
    available_models=[f for f in listdir(file_path) if f.startswith("model")]  
    logger.info(
        "Available models: %s", 
        str(available_models)
        )
    
    if len(available_models)>0:
        return joblib.load(file_path+max(available_models))
    else:
        logger.info("No model found.")

def get_s3_client(service_name):

    session = boto3.session.Session()

    return session.client(
        service_name=service_name,
        endpoint_url=config.s3_endpointurl
    )

def get_s3_resource(service_name):

    return boto3.resource(service_name,
            endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            use_ssl=True,
            verify=False
        )

def check_file_exist_s3(key, bucket):
    
    s3_resource = get_s3_resource("s3")
    
    s3_bucket=s3_resource.Bucket(bucket)

    for bucket_object in s3_bucket.objects.all():
        if key in bucket_object.key:
            return True
        else:
            return False

def sync_s3_bucket_to_artifacts(bucket,project_name=None):

    if project_name is None:
        project_name="kyt-cib"
    
    files_to_sync = []

    s3_resource = get_s3_resource("s3")
    
    s3_bucket = s3_resource.Bucket(bucket)
   
    for bucket_object in s3_bucket.objects.all():
        if project_name in bucket_object.key:
            files_to_sync.append(bucket_object.key)

    for f in files_to_sync:
        print("Downloading: ", f)

        directory, filename=os.path.split(f)
        path = Path(os.path.join(os.getcwd(),directory))

        print("Saving to: ", )
        path.mkdir(parents=True, exist_ok=True)
        get_s3_client("s3").download_file(
            Bucket=bucket,
            Key=f,
            Filename=os.path.join(path,filename)
        )
    return
    
def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = get_s3_client("s3")

    try:
        response = s3_client.upload_file(
            file_name, 
            bucket, 
            object_name
        )
    except ClientError as e:
        logger.error(e)
        return False
    
    return True

def sync_artifacts_to_s3_bucket(bucket, directory, if_exists="skip"):
    
    files_to_sync=[]

    for root, dirs, files in os.walk(directory):
        for filename in files:
            files_to_sync.append(
                (
                    os.path.join(root, filename),
                    "kyt-cib/"+os.path.join(root, filename)
                )
            )
    
    for f in files_to_sync:
        if if_exists=="skip":
            if not check_file_exist_s3(f[1], bucket):
                upload_file(f[0], bucket, f[1])
        elif if_exists=="replace":
            upload_file(f[0], bucket, f[1])

    return

def get_url(url, sep):
    url=urlparse(url)
    if url.scheme=="s3":
        client=get_s3_client("s3")
        obj=client.get_object(Bucket=url.netloc, Key=url.path[1:])
        return pd.read_csv(obj['Body'],sep=sep)
    else:
        return pd.read_csv(url,sep=sep)
    
