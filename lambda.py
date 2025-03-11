import json
import boto3
import pandas as pd
from io import StringIO
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    BUCKET_NAME = 'knowledgebase-bedrock-agent-ab3'

    logger.info(f"Event received: {json.dumps(event)}")

    # Initialize S3 client
    s3_client = boto3.client('s3')

    def get_named_parameter(event, name, default=None):
        """Safely get a named parameter from the event"""
        try:
            return next((item['value'] for item in event.get('parameters', [])
                        if item['name'] == name), default)
        except (KeyError, StopIteration):
            logger.warning(f"Parameter {name} not found in event")
            return default

    def read_s3_json(bucket, key):
        """Read JSONL data from S3"""
        try:
            logger.info(f"Reading JSON from s3://{bucket}/{key}")
            response = s3_client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read().decode('utf-8')
            # Split content by lines and parse each line as separate JSON
            json_objects = []
            for line in content.strip().split('\n'):
                if line:  # Skip empty lines
                    json_objects.append(json.loads(line))
            logger.info(f"Successfully read {len(json_objects)} JSON objects")
            return json_objects
        except Exception as e:
            logger.error(f"Error reading JSON from S3: {str(e)}")
            return None

    def read_s3_csv(bucket, key):
        """Read CSV data from S3"""
        try:
            logger.info(f"Reading CSV from s3://{bucket}/{key}")
            response = s3_client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read().decode('utf-8')
            df = pd.read_csv(StringIO(content))
            logger.info(f"Successfully read CSV with {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error reading CSV from S3: {str(e)}")
            return None

    def get_user_segment(segment_id):
        """Retrieve user segment data from S3"""
        segments = read_s3_json(
            BUCKET_NAME, 'segments/batch_segment_input_ab3.json.out')
        if not segments:
            logger.warning(f"No segments found")
            return []

        # Look for segment with matching itemId in input
        for segment in segments:
            if segment.get('input', {}).get('itemId') == segment_id:
                return segment.get('output', {}).get('usersList', [])

        # If no exact match, return the first segment (for demo purposes)
        if segments and 'output' in segments[0] and 'usersList' in segments[0]['output']:
            logger.info(f"Using first available segment as fallback")
            return segments[0]['output']['usersList']

        logger.warning(f"No valid segments found")
        return []

    def get_flight_details(item_id):
        """Get flight details from items data"""
        items_df = read_s3_csv(BUCKET_NAME, 'data/travel_items.csv')
        if items_df is None:
            logger.error("Failed to read items CSV")
            return None

        flight = items_df[items_df['ITEM_ID'] == item_id]
        if flight.empty:
            logger.warning(f"No flight found with ITEM_ID: {item_id}")
            return None

        return flight.iloc[0].to_dict()

    def get_user_tiers(user_ids):
        """Get user tiers for a list of users"""
        if not user_ids:
            logger.warning("Empty user_ids list provided")
            return {}

        users_df = read_s3_csv(BUCKET_NAME, 'data/travel_users.csv')
        if users_df is None:
            logger.error("Failed to read users CSV")
            return {}

        filtered_users = users_df[users_df['USER_ID'].isin(user_ids)]
        if filtered_users.empty:
            logger.warning("No matching users found in users CSV")
            return {}

        return filtered_users['MEMBER_TIER'].value_counts().to_dict()

    def generate_email_content(event):
        """Retrieve data for email content generation"""
        segment_id = get_named_parameter(event, 'segmentId')

        if not segment_id:
            logger.error("No segmentId provided in request")
            return {"error": "Missing required parameter: segmentId"}

        logger.info(f"Getting data for segment: {segment_id}")

        # Get user segment
        user_list = get_user_segment(segment_id)

        # Get flight details using segment_id as item_id
        flight_details = get_flight_details(segment_id)

        # If no flight details found with that ID, list available flights
        if not flight_details:
            items_df = read_s3_csv(BUCKET_NAME, 'data/travel_items.csv')
            if items_df is not None:
                available_flights = items_df[items_df['PROMOTION'] == 'Yes'].head(
                    5)[['ITEM_ID', 'SRC_CITY', 'DST_CITY', 'AIRLINE', 'MONTH']].to_dict('records')
                return {
                    "error": f"No flight details found for segment ID: {segment_id}",
                    "availablePromotions": available_flights
                }
            return {"error": f"No flight details found for segment ID: {segment_id}"}

        # Get user tier distribution
        user_tiers = get_user_tiers(user_list)

        # Return data for the agent to use in generating content
        return {
            "segmentId": segment_id,
            "userCount": len(user_list),
            "userTierDistribution": user_tiers,
            "flightDetails": {
                "itemId": flight_details.get('ITEM_ID'),
                "source": flight_details.get('SRC_CITY'),
                "destination": flight_details.get('DST_CITY'),
                "airline": flight_details.get('AIRLINE'),
                "duration": flight_details.get('DURATION_DAYS'),
                "price": flight_details.get('DYNAMIC_PRICE'),
                "promotion": flight_details.get('PROMOTION'),
                "month": flight_details.get('MONTH'),
                "discountForMembers": flight_details.get('DISCOUNT_FOR_MEMBER', 0)
            }
        }

    result = ''
    response_code = 200
    action_group = event.get('actionGroup', '')
    api_path = event.get('apiPath', '')

    logger.info(f"Processing request: {action_group}::{api_path}")

    try:
        if api_path == '/generateEmailContent':
            result = generate_email_content(event)
        else:
            response_code = 404
            result = {
                "error": f"Unrecognized api path: {action_group}::{api_path}"}
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        response_code = 500
        result = {"error": f"Internal server error: {str(e)}"}

    response_body = {
        'application/json': {
            'body': result
        }
    }

    action_response = {
        'actionGroup': event.get('actionGroup', ''),
        'apiPath': event.get('apiPath', ''),
        'httpMethod': event.get('httpMethod', ''),
        'httpStatusCode': response_code,
        'responseBody': response_body
    }

    api_response = {'messageVersion': '1.0', 'response': action_response}
    logger.info(f"Returning response: {json.dumps(api_response)}")
    return api_response
