import os
import logging

import boto3


class S3Helper:
    def __init__(self):
        AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
        AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')

        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        
        logging.info('S3 client created')
        
    def download(self, s3_bucket, s3_key, local_path):
        self.s3_client.download_file(s3_bucket, s3_key, local_path)
        
        return local_path
        
    def upload(self, local_path, s3_bucket, s3_key):
        self.s3_client.upload_file(local_path, s3_bucket, s3_key)
            
        return self.bucket_key_to_s3_uri(s3_bucket, s3_key)
        
    def bucket_key_to_s3_uri(self, bucket, key):
        return f's3://{bucket}/{key}'
            
    def s3_uri_to_bucket_key(self, s3_uri):
        s3_uri = s3_uri.replace('s3://', '')
        bucket, key = s3_uri.split('/', 1)
        return bucket, key
    
    def s3_uri_to_link(self, s3_uri):
        bucket, key = self.s3_uri_to_bucket_key(s3_uri)
        return f'https://{bucket}.s3.amazonaws.com/{key}'