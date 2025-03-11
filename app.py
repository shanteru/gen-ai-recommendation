import streamlit as st
import boto3
import pandas as pd
import json
import time
import os
from io import StringIO



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


def generate_email_with_agent(segment_id):
    session_id = f"demo-session-{int(time.time())}"
    
    # Get flight details for dynamic prompting
    flight_df = read_s3_csv(BUCKET_NAME, 'data/travel_items.csv')
    flight_details = None
    
    if flight_df is not None and segment_id:
        matching_flight = flight_df[flight_df['ITEM_ID'] == segment_id]
        if not matching_flight.empty:
            flight_details = matching_flight.iloc[0]
    
    # Create a specific, detailed prompt
    if flight_details is not None:
        prompt = f"""Generate a marketing email for a flight from {flight_details['SRC_CITY']} to {flight_details['DST_CITY']} with {flight_details['AIRLINE']} in {flight_details['MONTH']}. 
        
Price: ${flight_details['DYNAMIC_PRICE']}
Duration: {flight_details['DURATION_DAYS']} days
        
Create a compelling subject line and email body that highlights the attractions of {flight_details['DST_CITY']}. 
Include a call to action to book through our website https://demobooking.demo.co.
Format the email with emojis and bullet points for easy readability."""
    else:
        prompt = f"Generate a marketing email for flight segment ID {segment_id}. Include a subject line and compelling email body."
    
    # Create a fresh client for each call
    bedrock_agent_runtime = boto3.client(
        service_name='bedrock-agent-runtime',
        region_name='us-east-1'
    )
    
    # DEBUG: Log the client configuration
    st.info(f"Agent ID: {AGENT_ID}")
    st.info(f"Agent Alias ID: {AGENT_ALIAS_ID}")
    st.info(f"Session ID: {session_id}")
    
    try:
        # Make the API call
        response = bedrock_agent_runtime.invoke_agent(
            agentId=AGENT_ID,
            agentAliasId=AGENT_ALIAS_ID,
            sessionId=session_id,
            inputText=prompt
        )
        
        # Handle the event stream correctly
        full_response = ""
        
        # The completion field contains the EventStream object
        event_stream = response.get('completion')
        
        # Process the event stream if it exists
        if event_stream:
            for event in event_stream:
                # Check if the event has a 'chunk' attribute
                if hasattr(event, 'chunk'):
                    chunk = event.chunk
                    # Check if the chunk has a 'bytes' attribute
                    if hasattr(chunk, 'bytes'):
                        # Decode the bytes content to text
                        chunk_bytes = chunk.bytes
                        try:
                            chunk_text = chunk_bytes.decode('utf-8')
                            full_response += chunk_text
                        except:
                            st.error("Failed to decode chunk bytes")
                    # Or check if it has a message attribute
                    elif hasattr(chunk, 'message') and hasattr(chunk.message, 'content'):
                        for content_item in chunk.message.content:
                            if hasattr(content_item, 'text'):
                                full_response += content_item.text
        
        if full_response:
            return full_response
        
        # If we got response but no content from the event stream
        if 'chunk' in response:
            # Try to extract content from the response's chunk
            chunk = response.get('chunk', {})
            if 'bytes' in chunk:
                try:
                    return chunk['bytes'].decode('utf-8')
                except:
                    st.warning("Failed to decode response bytes")
            
        # At this point, we need to look at the full response structure
        st.warning("Could not extract text from response - examining response structure")
        st.json(str(response))
        
        return "Could not extract content from the agent response. Please check the logs for details."
        
    except Exception as e:
        st.error(f"Error invoking agent: {str(e)}")
        return f"Error: {str(e)}"

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
