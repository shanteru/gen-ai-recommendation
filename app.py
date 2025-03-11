import streamlit as st
import boto3
import pandas as pd
import json
import time
import os
from io import StringIO
from langchain_community.chat_models import BedrockChat


# Set up page config
st.set_page_config(
    page_title="Wanderly Email Campaign Generator",
    page_icon="✈️",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    /* Text and heading colors - for dark mode */
    .main-header {
        font-size: 2.5rem;
        color: #39C0F0 !important;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #39C0F0 !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #39C0F0 !important;
    }
    p, li, label, div {
        color: white !important;
    }
    .st-bx {
        color: white !important;
    }
    
    /* Box styles */
    .success-box {
        padding: 1rem;
        background-color: rgba(46, 125, 50, 0.2);
        border-radius: 0.5rem;
        border-left: 0.5rem solid #4CAF50;
        color: white !important;
    }
    .info-box {
        padding: 1rem;
        background-color: rgba(30, 136, 229, 0.2);
        border-radius: 0.5rem;
        border-left: 0.5rem solid #2196F3;
        color: white !important;
    }
    
    /* Button styles */
    .stButton button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
    
    /* Tab styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(30, 136, 229, 0.1);
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: white !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5;
        color: white !important;
    }
    
    /* Input field text */
    .stTextInput input {
        color: white !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Table text */
    .dataframe {
        color: white !important;
    }
    .dataframe th {
        color: white !important;
        background-color: rgba(30, 136, 229, 0.3) !important;
    }
    .dataframe td {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Set up AWS clients


@st.cache_resource
def get_aws_clients():
    try:
        # First, try to use environment variables or instance profile
        s3_client = boto3.client('s3')

        # Check if Bedrock agent is available in the region
        bedrock_regions = boto3.Session().get_available_regions('bedrock-agent-runtime')
        current_region = boto3.Session().region_name

        # If the current region doesn't support Bedrock, use us-east-1
        bedrock_region = current_region if current_region in bedrock_regions else 'us-east-1'

        bedrock_agent_client = boto3.client(
            'bedrock-agent-runtime',
            region_name=bedrock_region
        )

        return s3_client, bedrock_agent_client
    except Exception as e:
        st.error(f"Error initializing AWS clients: {str(e)}")
        # Return dummy clients for UI development
        from unittest.mock import MagicMock
        return MagicMock(), MagicMock()


try:
    s3_client, bedrock_agent_client = get_aws_clients()
except Exception as e:
    st.error(f"Failed to initialize AWS clients: {str(e)}")
    # Create placeholder clients for UI testing
    from unittest.mock import MagicMock
    s3_client, bedrock_agent_client = MagicMock(), MagicMock()

# Configuration
BUCKET_NAME = 'knowledgebase-bedrock-agent-ab3'
AGENT_ID = os.environ.get('AGENT_ID', '')
AGENT_ALIAS_ID = os.environ.get('AGENT_ALIAS_ID', 'TSTALIASID')

# Helper functions to read data


@st.cache_data(ttl=300)
def read_s3_csv(bucket, key):
    """Read CSV data from S3"""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(content))
        return df
    except Exception as e:
        st.error(f"Error reading CSV from S3: {str(e)}")
        return None


@st.cache_data(ttl=300)
def read_s3_json(bucket, key):
    """Read JSONL data from S3"""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        json_objects = []
        for line in content.strip().split('\n'):
            if line:  # Skip empty lines
                json_objects.append(json.loads(line))
        return json_objects
    except Exception as e:
        st.error(f"Error reading JSON from S3: {str(e)}")
        return None

# Function to interact with Bedrock Agent


def generate_email_with_agent(segment_id, customization=""):
    """Generate email content from Bedrock Agent and extract just the clean email text"""
    session_id = f"demo-session-{int(time.time())}"

    # Get flight details from session state for mock response
    flight_details = st.session_state.get('flight_details', {})
    src_city = flight_details.get('SRC_CITY', 'Unknown City')
    dst_city = flight_details.get('DST_CITY', 'Unknown City')
    airline = flight_details.get('AIRLINE', 'Unknown Airline')
    month = flight_details.get('MONTH', 'Unknown Month')
    price = flight_details.get('DYNAMIC_PRICE', 9999)
    duration = flight_details.get('DURATION_DAYS', 10)

    # Create dynamic mock response using flight details
    mock_response = f"""Subject: Exclusive Deal: Fly from {src_city} to {dst_city} This {month}!

Dear Valued Wanderly Traveler,

We're excited to offer you an exclusive opportunity to explore the vibrant city of {dst_city} this {month}! 

✈️ {src_city} to {dst_city}
🗓️ Travel Period: {month} 2023
💰 Special Price: ${price:,}
⭐ Duration: {duration} days
🛫 Airline: {airline}

During your stay, you might enjoy:
- Exploring the local culture and attractions
- Discovering the city's hidden gems
- Experiencing the local cuisine
- Creating unforgettable memories

Book now at https://demobooking.demo.co and use promo code {dst_city.upper().replace(' ', '')}23 to secure this special offer!

Best regards,
The Wanderly Team"""

    # Create chat model for token counting
    chat_model = BedrockChat(
        client=bedrock_agent_client,
        model_id="anthropic.claude-3-haiku-20240307-v1:0"
    )

    # Build input text with customization
    input_text = f"Generate an email marketing campaign for flight segment ID {segment_id}"
    if customization.strip():
        input_text += f"\n\nCustomization preferences:\n{customization}"

    # Count input tokens
    input_tokens = chat_model.get_num_tokens(input_text)

    try:
        # Check if Agent ID and Alias ID are provided
        if not AGENT_ID or AGENT_ID == '':
            st.warning(
                "Agent ID is not configured. Using mock response for demo purposes.")
            mock_tokens = chat_model.get_num_tokens(mock_response)
            return {
                'content': mock_response,
                'tokens': {
                    'input': input_tokens,
                    'output': mock_tokens,
                    'total': input_tokens + mock_tokens
                }
            }

        # Invoke the agent
        response = bedrock_agent_client.invoke_agent(
            agentId=AGENT_ID,
            agentAliasId=AGENT_ALIAS_ID,
            sessionId=session_id,
            inputText=input_text,
            enableTrace=True
        )

        # Process the response
        if 'completion' in response:
            event_stream = response['completion']
            raw_content = ""
            email_content = ""

            # Look for chunks that contain the bytes field - these have the actual email content
            for event in event_stream:
                if 'chunk' in event and 'bytes' in event['chunk']:
                    # Extract the bytes and decode to text
                    try:
                        # This is the most reliable way - the bytes field contains the actual email
                        content_bytes = event['chunk']['bytes']
                        if isinstance(content_bytes, bytes):
                            decoded = content_bytes.decode('utf-8')
                            email_content += decoded
                    except Exception as e:
                        st.warning(f"Error decoding bytes: {str(e)}")
                        # Fall back to string representation
                        raw_content += str(event)
                else:
                    # Collect other content for fallback
                    raw_content += str(event)

            # If we found email content in the bytes field
            if email_content:
                # Clean up the email content
                # 1. Remove the analysis part at the end (which starts with double newlines)
                email_parts = email_content.split("\n\n\n")
                if len(email_parts) > 1:
                    # The first part is the actual email
                    clean_email = email_parts[0].strip()
                else:
                    clean_email = email_content.strip()

                # 2. Format special tags like <call_to_action> to look better
                import re
                # Replace XML-like tags with formatted text
                clean_email = re.sub(r'<call_to_action>(.*?)</call_to_action>',
                                     r'\n--- Call to Action ---\n\1\n-------------------\n',
                                     clean_email)

                # 3. Replace any other placeholder-like elements
                clean_email = clean_email.replace(
                    "[Customer]", "Valued Customer")
                clean_email = clean_email.replace(
                    "[Member/Gold]", "Valued Member")
                clean_email = clean_email.replace("[Book Now]", "BOOK NOW")

                # Get token counts
                output_tokens = chat_model.get_num_tokens(clean_email)

                return {
                    'content': clean_email,
                    'tokens': {
                        'input': input_tokens,
                        'output': output_tokens,
                        'total': input_tokens + output_tokens
                    }
                }

            # If we didn't get email content from bytes, try to extract from raw content
            if raw_content:
                import re
                # Look for the email content pattern - starting with "Subject:" and ending before analysis
                email_match = re.search(r'Subject:.*?(?:Best regards,|Sincerely,|Wishing you|The Wanderly Team)[^\n]*',
                                        raw_content, re.DOTALL)
                if email_match:
                    clean_email = email_match.group(0).strip()
                    output_tokens = chat_model.get_num_tokens(clean_email)
                    return {
                        'content': clean_email,
                        'tokens': {
                            'input': input_tokens,
                            'output': output_tokens,
                            'total': input_tokens + output_tokens
                        }
                    }

                # If still not found, try to extract any textResponsePart with Subject
                text_parts = re.findall(
                    r'"text"\s*:\s*"(Subject:.*?)"', raw_content)
                if text_parts:
                    # Unescape and clean up the text
                    combined = ' '.join(text_parts)
                    combined = combined.replace('\\n', '\n')
                    output_tokens = chat_model.get_num_tokens(combined)
                    return {
                        'content': combined,
                        'tokens': {
                            'input': input_tokens,
                            'output': output_tokens,
                            'total': input_tokens + output_tokens
                        }
                    }

            # If all extraction attempts fail, return mock response
            mock_tokens = chat_model.get_num_tokens(mock_response)
            return {
                'content': mock_response,
                'tokens': {
                    'input': input_tokens,
                    'output': mock_tokens,
                    'total': input_tokens + mock_tokens
                }
            }

        else:
            # No completion in response
            mock_tokens = chat_model.get_num_tokens(mock_response)
            return {
                'content': mock_response,
                'tokens': {
                    'input': input_tokens,
                    'output': mock_tokens,
                    'total': input_tokens + mock_tokens
                }
            }

    except Exception as e:
        st.error(f"Error generating email: {str(e)}")
        mock_tokens = chat_model.get_num_tokens(mock_response)
        return {
            'content': mock_response,
            'tokens': {
                'input': input_tokens,
                'output': mock_tokens,
                'total': input_tokens + mock_tokens
            }
        }


# Title
st.markdown('<p class="main-header">Wanderly Email Campaign Generator</p>',
            unsafe_allow_html=True)
st.markdown("Generate personalized marketing emails for flight promotions")
st.markdown("---")

# Create tabs for different steps - only 2 tabs now
tab1, tab2 = st.tabs(["📋 Select Flight & View Segment", "✉️ Generate Email"])

with tab1:
    st.markdown('<p class="sub-header">1. Select Flight to Promote</p>',
                unsafe_allow_html=True)

    # Load flight data
    flight_df = read_s3_csv(BUCKET_NAME, 'data/travel_items.csv')

    if flight_df is not None:
        # Filter to only show March promotions as that's what we have segments for
        promo_flights = flight_df[(flight_df['PROMOTION'] == 'Yes') & (
            flight_df['MONTH'] == 'March')]

        if promo_flights.empty:
            # If no March promotions, show all promotions as fallback
            promo_flights = flight_df[flight_df['PROMOTION'] == 'Yes']
            st.warning(
                "Note: Segmentation data is only available for March promotions. Other months shown as reference.")

        if promo_flights.empty:
            st.warning("No promotional flights found.")
        else:
            # Display flights as a table
            st.markdown(
                '<div class="info-box">Select one of the available promotional flights below:</div>', unsafe_allow_html=True)

            # Display table with flight IDs
            st.dataframe(
                promo_flights[['ITEM_ID', 'SRC_CITY', 'DST_CITY', 'AIRLINE',
                               'MONTH', 'DYNAMIC_PRICE']].reset_index(drop=True),
                column_config={
                    "ITEM_ID": "Flight ID",
                    "SRC_CITY": "From",
                    "DST_CITY": "To",
                    "AIRLINE": "Airline",
                    "MONTH": "Month",
                    "DYNAMIC_PRICE": st.column_config.NumberColumn("Price", format="$%d")
                },
                use_container_width=True
            )

            # Option to directly copy-paste Flight ID
            selected_flight_id = st.text_input(
                "Enter Flight ID (copy from table above):",
                help="Copy the Flight ID directly from the table above"
            )

            # Button to process the flight ID
            if st.button("Analyze User Segment", use_container_width=True):
                if selected_flight_id.strip() == "":
                    st.error("Please enter a Flight ID first")
                else:
                    # Check if flight ID exists
                    matching_flight = promo_flights[promo_flights['ITEM_ID']
                                                    == selected_flight_id]

                    if matching_flight.empty:
                        st.error(
                            f"Flight ID {selected_flight_id} not found in promotional flights")
                    else:
                        st.session_state['selected_flight_id'] = selected_flight_id
                        st.session_state['flight_details'] = matching_flight.iloc[0]

                        # Show flight details and segment information on the same page
                        st.markdown("---")
                        st.markdown(
                            '<p class="sub-header">2. User Segment Details</p>', unsafe_allow_html=True)

                        flight_details = matching_flight.iloc[0]

                        st.markdown(f"""
                        <div class="info-box">
                        Analyzing user segment for flight: {flight_details['SRC_CITY']} to {flight_details['DST_CITY']} ({flight_details['AIRLINE']})
                        </div>
                        """, unsafe_allow_html=True)

                        # Load segment data
                        segments = read_s3_json(
                            BUCKET_NAME, 'segments/batch_segment_input_ab3.json.out')

                        if segments:
                            # Find matching segment
                            matching_segment = None
                            for segment in segments:
                                if segment.get('input', {}).get('itemId') == selected_flight_id:
                                    matching_segment = segment
                                    break

                            if matching_segment:
                                user_list = matching_segment.get(
                                    'output', {}).get('usersList', [])
                                st.session_state['user_list'] = user_list

                                if user_list:
                                    st.markdown(
                                        f'<div class="success-box">✅ Found {len(user_list)} users in this segment!</div>', unsafe_allow_html=True)

                                    # Load user data to show distribution
                                    user_df = read_s3_csv(
                                        BUCKET_NAME, 'data/travel_users.csv')
                                    if user_df is not None:
                                        segment_users = user_df[user_df['USER_ID'].isin(
                                            user_list)]
                                        tier_counts = segment_users['MEMBER_TIER'].value_counts(
                                        )

                                        col1, col2 = st.columns(2)

                                        with col1:
                                            # Display as pie chart
                                            st.subheader(
                                                "User Tier Distribution")
                                            st.bar_chart(tier_counts)

                                        with col2:
                                            st.subheader("User Demographics")
                                            st.markdown(f"""
                                            - **Total users:** {len(user_list)}
                                            - **Gold members:** {tier_counts.get('Gold', 0)}
                                            - **Silver members:** {tier_counts.get('Silver', 0)}
                                            - **Regular members:** {tier_counts.get('Member', 0)}
                                            """)

                                        st.markdown(
                                            '<div class="info-box">Now you can go to the "Generate Email" tab to create an email campaign for this segment</div>', unsafe_allow_html=True)

                                        # Show a sample of user IDs
                                        with st.expander("View Sample User IDs"):
                                            st.write(user_list[:10])
                                            st.info(
                                                f"Showing 10 of {len(user_list)} users")
                                    else:
                                        st.error("Could not load user data")
                                else:
                                    st.warning(
                                        "No users found in this segment")
                            else:
                                # No exact match found, show all available segments
                                st.warning(
                                    f"No exact segment match found for flight ID: {selected_flight_id}")

                                # Display all available segments as an alternative
                                segment_options = []
                                for segment in segments:
                                    item_id = segment.get(
                                        'input', {}).get('itemId')
                                    user_count = len(segment.get(
                                        'output', {}).get('usersList', []))
                                    segment_options.append(
                                        {"item_id": item_id, "user_count": user_count})

                                st.markdown("Available segments:")
                                for option in segment_options:
                                    if option["item_id"] in flight_df['ITEM_ID'].values:
                                        flight_info = flight_df[flight_df['ITEM_ID']
                                                                == option["item_id"]].iloc[0]
                                        st.markdown(
                                            f"- {flight_info['SRC_CITY']} to {flight_info['DST_CITY']} - {option['user_count']} users (ID: {option['item_id']})")
                        else:
                            st.error("No segment data available.")
    else:
        st.error("Could not load flight data.")

with tab2:
    st.markdown('<p class="sub-header">Generate Email Campaign</p>',
                unsafe_allow_html=True)

    if 'selected_flight_id' not in st.session_state or 'flight_details' not in st.session_state:
        st.info(
            "Please select a flight and analyze a user segment first (in the previous tab)")
    else:
        selected_flight_id = st.session_state['selected_flight_id']
        flight_details = st.session_state['flight_details']
        user_list = st.session_state.get('user_list', [])

        st.markdown(f"""
        <div class="info-box">
        Ready to generate email campaign for {flight_details['SRC_CITY']} to {flight_details['DST_CITY']} targeting {len(user_list)} users
        </div>
        """, unsafe_allow_html=True)

        flight_id_input = st.text_input(
            "Flight ID:",
            value=selected_flight_id,
            disabled=True
        )

        # Add customization input right before the generate button
        customization = st.text_area(
            "Email Customization (Optional)",
            placeholder="""Specify your preferences for the email:
- Tone (formal/casual/friendly)
- Key points to emphasize
- Special features to highlight
- Any specific content to include""",
            help="Your preferences will be used to customize the email content"
        )

        if st.button("Generate Personalized Email", use_container_width=True):
            with st.spinner("Generating personalized email content..."):
                # Call the Bedrock Agent with customization
                result = generate_email_with_agent(
                    selected_flight_id, customization)
                email_content = result['content']
                token_counts = result['tokens']

                if email_content:
                    st.session_state['generated_email'] = email_content

                    # Your existing subject line extraction code...
                    # Keep all the existing email display code...

                    # Add token usage metrics after the email preview
                    st.markdown("### Token Usage")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Input Tokens", token_counts['input'])
                    with col2:
                        st.metric("Output Tokens", token_counts['output'])
                    with col3:
                        st.metric("Total Tokens", token_counts['total'])

                    # Keep your existing download buttons and user list code...

        if st.button("Generate Personalized Email", use_container_width=True):
            with st.spinner("Generating personalized email content..."):
                # Call the Bedrock Agent
                email_content = generate_email_with_agent(selected_flight_id)

                if email_content:
                    st.session_state['generated_email'] = email_content

                    # Try to extract subject line using common patterns
                    subject_line = ""
                    if "Subject:" in email_content:
                        subject_line = email_content.split(
                            "Subject:")[1].split("\n")[0].strip()
                    elif "SUBJECT:" in email_content:
                        subject_line = email_content.split(
                            "SUBJECT:")[1].split("\n")[0].strip()

                    st.markdown(
                        f'<div class="success-box">✅ Email campaign generated successfully!</div>', unsafe_allow_html=True)

                    # Show email preview
                    st.subheader("Email Preview")

                    # Display in a nice format with tabs for different views
                    preview_tab1, preview_tab2 = st.tabs(
                        ["Formatted View", "Raw Text"])

                    with preview_tab1:
                        # Try to extract and format the email nicely
                        if subject_line:
                            st.markdown(f"### {subject_line}")

                        # Format the body - replace newlines with HTML breaks for better display
                        email_body = email_content
                        if subject_line:
                            # Remove the subject line from the body if we extracted it
                            email_body = email_body.replace(
                                f"Subject: {subject_line}", "")
                            email_body = email_body.replace(
                                f"SUBJECT: {subject_line}", "")

                        # Convert markdown-like formatting to actual markdown
                        email_body = email_body.replace(
                            "**", "**").replace("__", "**")

                        st.markdown(email_body)

                    with preview_tab2:
                        # Show the raw text
                        st.text_area("Email Content",
                                     email_content, height=400)

                    # Download button
                    st.download_button(
                        label="Download Email Content",
                        data=email_content,
                        file_name=f"wanderly_campaign_{flight_details['DST_CITY'].lower().replace(' ', '_')}.txt",
                        mime="text/plain"
                    )

                    # User list download
                    st.subheader("Target User List")
                    st.markdown(
                        f"This campaign will target **{len(user_list)}** users")

                    # Convert user list to CSV for download
                    user_csv = "USER_ID\n" + "\n".join(user_list)
                    st.download_button(
                        label="Download User List (CSV)",
                        data=user_csv,
                        file_name=f"wanderly_users_{flight_details['DST_CITY'].lower().replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.error(
                        "Failed to generate email content. Please try again.")

        # Display previously generated email if available
        if 'generated_email' in st.session_state:
            st.subheader("Previously Generated Email")
            st.text_area("Email Content",
                         st.session_state['generated_email'], height=300)
