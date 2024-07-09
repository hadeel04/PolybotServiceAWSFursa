import json
from collections import Counter
import flask
from flask import request
import os
from bot import ObjectDetectionBot
import boto3
from botocore.exceptions import ClientError

app = flask.Flask(__name__)

REGION = os.environ['REGION']
DYNAMODB_TABLE_NAME = os.environ['DYNAMODB_TABLE_NAME']
# TODO load TELEGRAM_TOKEN value from Secret Manager

def get_secret():
    secret_name = "hadeel-secret"
    region_name = REGION

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e
    secret_string = get_secret_value_response['SecretString']
    secret_dict = json.loads(secret_string)
    TELEGRAM_TOKEN = secret_dict.get('TELEGRAM_TOKEN')
    return TELEGRAM_TOKEN


TELEGRAM_TOKEN = get_secret()
TELEGRAM_APP_URL = os.environ['TELEGRAM_APP_URL']


@app.route('/', methods=['GET'])
def index():
    return 'Ok'


@app.route(f'/{TELEGRAM_TOKEN}/', methods=['POST'])
def webhook():
    req = request.get_json()
    bot.handle_message(req['message'])
    return 'Ok'


def get_emoji_for_class(class_name):
    emoji_map = {
        "person": "👤", "car": "🚗", "dog": "🐕", "cat": "🐈",
        "bird": "🐦", "airplane": "✈️", "bicycle": "🚲", "boat": "🚢",
        # Add more mappings as needed
    }
    return emoji_map.get(class_name.lower(), "")


def format_detection_results(labels):
    # Count occurrences of each class
    class_counts = Counter(label['class'] for label in labels)

    # Sort classes by count, then alphabetically
    sorted_classes = sorted(class_counts.items(), key=lambda x: (-x[1], x[0]))

    # Format the results
    text_results = "📸 Detection Results:\n\n"
    for class_name, count in sorted_classes:
        emoji = get_emoji_for_class(class_name)
        if emoji:
            prefix = f"{emoji} "
        else:
            prefix = ""

        if count == 1:
            text_results += f"{prefix}{class_name}\n"
        else:
            text_results += f"{prefix}{class_name} (x{count})\n"

    total_objects = sum(class_counts.values())
    text_results += f"\nTotal objects detected: {total_objects}"

    return text_results


@app.route(f'/results', methods=['POST'])
def results():
    prediction_id = request.args.get('predictionId')

    # TODO use the prediction_id to retrieve results from DynamoDB and send to the end-user
    dynamodb = boto3.resource('dynamodb', region_name=REGION)
    table = dynamodb.Table(DYNAMODB_TABLE_NAME)
    response = table.get_item(Key={'prediction_id': prediction_id})
    prediction = response['Item']
    chat_id = prediction['chat_id']
    labels = prediction['labels']

    text_results = format_detection_results(labels)
    bot.send_text(chat_id, text_results)
    return 'Ok'


@app.route(f'/loadTest/', methods=['POST'])
def load_test():
    req = request.get_json()
    bot.handle_message(req['message'])
    return 'Ok'


if __name__ == "__main__":
    bot = ObjectDetectionBot(TELEGRAM_TOKEN, TELEGRAM_APP_URL)
    ssl_context = ('cert.pem', 'key.key')
    app.run(host='0.0.0.0', port=8443, ssl_context=ssl_context)
