import requests
import boto3
import pytz
import json
import os
import numpy as np
import cv2
import pickle 

from datetime import datetime, timedelta

s3 = boto3.client('s3',
    aws_access_key_id="AKIA2DGCWL5MLHSGP26A",#os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key="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"#os.getenv("AWS_SECRET_KEY")
)

TODAY_S3 = datetime.now(pytz.timezone('Singapore')).strftime("%Y-%m-%d")
YESTERDAY_S3 = (datetime.now(pytz.timezone('Singapore')) - timedelta(days = 1)).strftime("%Y-%m-%d")

DS_BUCKET = "mytukar-data-science"
IMAGE_BUCKET = "mytukar"
ROOT_DIR = 'registrationcard/'
PREDICTION_DIR = f"{ROOT_DIR}prediction/"
SUBMISSION_DIR= f"{ROOT_DIR}submission/"

LABEL_DIR= f"{ROOT_DIR}label/"

TRAINING_DIR= f"{ROOT_DIR}training/"
HISTORY_DIR = f"{TRAINING_DIR}history/"
MODEL_DIR = f"{TRAINING_DIR}model/"

def coder_to_s3(coder, filename):
    """
    Store encoder or decoder to s3
    Params:
        coder: encoder or decoder in json format
    Returns:
        key: stored key on s3
    """
    key = f"{LABEL_DIR}{filename}"
    print(f"Encoder/decoder has successfully been saved!")
    s3.put_object(Bucket=DS_BUCKET,Key=key, Body=json.dumps(coder, indent=2))
    return key


def coder_from_s3(filename):
    """
    Retrieve encoder or decoder from s3
    Params:
        filename: encoder or decoder filename
    Returns:
        coder: encoder or decoder in json format
    """
    decoder_key = f"{LABEL_DIR}{filename}"
    try:
        obj = s3.get_object(Bucket=DS_BUCKET, Key=decoder_key)
        decoded_obj = obj['Body'].read().decode('utf-8')
        coder = json.loads(decoded_obj)
        print(f"Retrieved coder from s3 {decoder_key}")
        return coder
    except Exception as e:
        print(f"Unexpected error on downloading coder: {filename} {e}")

def weights_to_s3(weights, filename="recognizer-weights"):
    """
    Stored weights to s3
        Params:
            weights: weights in np array
    """
    weights_to_s3_serialized = pickle.dumps(weights,protocol=-1)
    weights_key = MODEL_DIR+filename
    try:
        s3.put_object(Bucket=DS_BUCKET,Key=weights_key, Body=weights_to_s3_serialized)
        print(f'New model saved - {weights_key}')

    except Exception as e:
        print(f"Unexpected error on uploading model: {filename} {e}")

def weights_from_s3(filename="recognizer-weights"):
    """
    Retrieve weights from s3
        Params:
            filename: weights filename
        Returns:
            weights: weights in np array
    """   
    weights_key = MODEL_DIR+filename
    try:
        obj = s3.get_object(Bucket=DS_BUCKET, Key=weights_key)
        decoded_obj = obj['Body'].read()
        weights = pickle.loads(decoded_obj)
        return weights
    except Exception as e:
        print(f"Unexpected error on downloading model: {filename} {e}")

def history_to_s3(history):
    """
    Store history of training to S3 bucket
        Params:
            historay: A dictionary containing list of loss (error gap) and accuracy during training
        Returns:
            key: history filepath stored in s3
    """
    key = f"{HISTORY_DIR}{TODAY_S3}-history"
    history_serialized = json.dumps(str(history), indent=2)
    s3.put_object(Bucket=DS_BUCKET,Key=key, Body=history_serialized)
    print(f"Training history for {TODAY_S3} has successfully been saved!")

    return key

def get_corrected_filenames():
    """
    Retrieve yesterday submission 
        Returns:
            filenames: List of submission filenames
    """
    response = s3.list_objects_v2(
            Bucket=DS_BUCKET,
            Prefix =f"{SUBMISSION_DIR}{YESTERDAY_S3}/",
            MaxKeys=200 )
    if 'Contents' in response:
        filenames = [ content['Key'] for content in response['Contents']]
        return filenames
    return []

def get_corrections():
    """
    Retrieve yesterday correction for re-train purposes
        Returns:
            img_name: Image filename
            img: image, basically a 3 dimension np array
            prediction: prediction in dictionary format containing , filename, text, and boxes (xy coordinates)
            submission: submission result in dictionary format containing submitted no_pendaftaran, no_chasis, no_enjin
    """
    filenames = get_corrected_filenames()
    try:
        corrections = []
        for filename in filenames:
            img_name = filename.split('/')[-1]
            img = img_from_s3(img_name)

            submission_key= filename
            submission_obj = s3.get_object(Bucket=DS_BUCKET, Key=submission_key)
            submission_buffer = submission_obj['Body'].read().decode('utf-8')
            submission = json.loads(submission_buffer)

            prediction_key = filename.replace('submission','prediction')
            prediction_obj = s3.get_object(Bucket=DS_BUCKET, Key=prediction_key)
            prediction_buffer = prediction_obj['Body'].read().decode('utf-8')
            prediction = json.loads(prediction_buffer)
            
            corrections.append((img_name, img, prediction, submission))
        return corrections
    except Exception as e:
        print(e)


def img_from_s3(filename):
    """
    Retrieve images stored in S3 and convert them to readable format
        Params:
            filename: Image filename without the format eg: jpg, png
        Returns:
            image: image, basically a 3 dimension np array
    """
    file_obj = s3.get_object(Bucket=IMAGE_BUCKET, Key=filename) 
    file_content = file_obj["Body"].read()

    np_array = np.frombuffer(file_content, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # print(f'image type: {type(image)}')
    image = image[:, :, :3]  #remove alpha from png

    return image

