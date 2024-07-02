import json
import time
from pathlib import Path
from detect import run
import yaml
from loguru import logger
import os
import boto3
import requests
from decimal import Decimal

images_bucket = os.environ['BUCKET_NAME']
queue_name = os.environ['SQS_QUEUE_NAME']

sqs_client = boto3.client('sqs', region_name='us-east-2')

with open("data/coco128.yaml", "r") as stream:
    names = yaml.safe_load(stream)['names']


def consume():
    while True:
        response = sqs_client.receive_message(QueueUrl=queue_name, MaxNumberOfMessages=1, WaitTimeSeconds=5)

        if 'Messages' in response:
            message = response['Messages'][0]['Body']
            receipt_handle = response['Messages'][0]['ReceiptHandle']

            # Use the ReceiptHandle as a prediction UUID
            prediction_id = response['Messages'][0]['MessageId']

            logger.info(f'prediction: {prediction_id}. start processing')

            # Receives a URL parameter representing the image to download from S3
            message_body = json.loads(message)
            img_name = message_body['img_name']
            chat_id = message_body['chat_id']

            # TODO download img_name from S3, store the local image path in original_img_path
            s3_client = boto3.client('s3')
            local_img_dir = 'tempImages'
            os.makedirs(local_img_dir, exist_ok=True)
            original_img_path = os.path.join(local_img_dir, img_name)
            s3_client.download_file(images_bucket, img_name, original_img_path)

            logger.info(f'prediction: {prediction_id}/{original_img_path}. Download img completed')

            # Predicts the objects in the image
            run(
                weights='yolov5s.pt',
                data='data/coco128.yaml',
                source=original_img_path,
                project='static/data',
                name=prediction_id,
                save_txt=True
            )

            logger.info(f'prediction: {prediction_id}/{original_img_path}. done')

            # This is the path for the predicted image with labels
            # The predicted image typically includes bounding boxes drawn around the detected objects, along with class labels and possibly confidence scores.
            #predicted_img_path = Path(f'static/data/{prediction_id}/{original_img_path}')
            predicted_img_path = Path(f'static/data/{prediction_id}/{img_name}')

            # TODO Uploads the predicted image (predicted_img_path) to S3 (be careful not to override the original image).
            predicted_img_s3_path = f'predictions/{prediction_id}/{img_name}'
            s3_client.upload_file(str(predicted_img_path), images_bucket, predicted_img_s3_path)

            # Parse prediction labels and create a summary
            #pred_summary_path = Path(f'static/data/{prediction_id}/labels/{original_img_path.split(".")[0]}.txt')
            pred_summary_path = Path(f'static/data/{prediction_id}/labels/{img_name.split(".")[0]}.txt')
            if pred_summary_path.exists():
                with open(pred_summary_path) as f:
                    labels = f.read().splitlines()
                    labels = [line.split(' ') for line in labels]
                    labels = [{
                        'class': names[int(l[0])],
                        'cx': Decimal(str(l[1])),
                        'cy': Decimal(str(l[2])),
                        'width': Decimal(str(l[3])),
                        'height': Decimal(str(l[4])),
                    } for l in labels]

                logger.info(f'prediction: {prediction_id}/{original_img_path}. prediction summary:\n\n{labels}')

                prediction_summary = {
                    'prediction_id': prediction_id,
                    'original_img_path': str(original_img_path),
                    'predicted_img_path': str(predicted_img_path),
                    'labels': labels,
                    'chat_id': chat_id,
                    'time': Decimal(str(time.time()))
                }

                # TODO store the prediction_summary in a DynamoDB table
                dynamodb = boto3.resource('dynamodb', region_name='us-east-2')
                table = dynamodb.Table('hadeel-yolo5-predictions')
                table.put_item(Item=prediction_summary)

                # TODO perform a GET request to Polybot to `/results` endpoint
                polybot_url = os.environ['POLYBOT_URL']
                requests.post(f'{polybot_url}/results', params={'predictionId': prediction_id})

            # Delete the message from the queue as the job is considered as DONE
            sqs_client.delete_message(QueueUrl=queue_name, ReceiptHandle=receipt_handle)


if __name__ == "__main__":
    consume()
