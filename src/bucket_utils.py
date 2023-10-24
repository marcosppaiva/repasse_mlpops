import os
from typing import Any

import boto3
from boto3.exceptions import S3UploadFailedError
from botocore.exceptions import ParamValidationError
from dotenv import load_dotenv

load_dotenv()

DATABASE_URI = os.getenv("DATABASE_URI", "")
POSTGRES_USER = os.getenv("POSTGRES_USER", "")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET", "")


def upload_file(filename: str, key: str, bucket: str = AWS_S3_BUCKET):
    try:
        s3_client = boto3.client("s3")

        s3_client.upload_file(Filename=filename, Bucket=bucket, Key=key)
    except (ParamValidationError, S3UploadFailedError) as error:
        raise error


def download_file(filename: str, key: str, bucket: str = AWS_S3_BUCKET):
    try:
        s3_client = boto3.client("s3")

        s3_client.download_file(
            Bucket=bucket,
            Key=key,
            Filename=filename,
        )
    except ParamValidationError as error:
        raise error


def download_obj(bucket_name: str, key: str) -> Any:
    try:
        s3_resource = boto3.resource("s3")

        s3_object = s3_resource.Object(bucket_name=bucket_name, key=key)

        s3_response = s3_object.get()

        s3_object_body = s3_response.get("Body")

        return s3_object_body
    except ParamValidationError as error:
        raise error
