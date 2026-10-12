import os
from pathlib import Path

import boto3
import psycopg
from psycopg.rows import dict_row


def secret(name):
    path = os.environ.get(name + '_FILE')
    return Path(path).read_text().strip() if path else os.environ[name]


def connect():
    return psycopg.connect(secret('ANNOTATION_DATABASE_URL'), row_factory=dict_row)


def s3():
    return boto3.client('s3', endpoint_url=os.environ['S3_ENDPOINT'],
                       aws_access_key_id=secret('S3_ACCESS_KEY'),
                       aws_secret_access_key=secret('S3_SECRET_KEY'),
                       region_name='us-east-1', verify=os.environ.get('S3_CA_BUNDLE') or True)


def bucket():
    return os.environ.get('S3_BUCKET', 'speech-data')


def dataset_bucket():
    return os.environ.get('S3_DATASET_BUCKET', bucket())


def initialize():
    with connect() as conn:
        conn.execute(Path(__file__).with_name('schema.sql').read_text())
    client = s3()
    try:
        client.head_bucket(Bucket=bucket())
    except client.exceptions.ClientError as error:
        if error.response['ResponseMetadata']['HTTPStatusCode'] != 404:
            raise
        client.create_bucket(Bucket=bucket())
