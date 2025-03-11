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
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .success-box {
        padding: 1rem;
        background-color: #E8F5E9;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #4CAF50;
    }
    .info-box {
        padding: 1rem;
        background-color: #E3F2FD;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #2196F3;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Set up AWS clients
@st.cache_resource
def get_aws_clients():
    s3_client = boto3.client('s3')
    bedrock_agent_client = boto3.client('bedrock-agent-runtime')
    return s3_client, bedrock_agent_client

s3_client, bedrock_agent_client = get_aws_clients()

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
    try:
        # First, ask the agent to generate an email for the segment
        response = bedrock_agent_client.invoke_agent(
            agentId=AGENT_ID,
            agentAliasId=AGENT_ALIAS_ID,
            sessionId=session_id,
            inputText=f"Generate an email marketing campaign for flight segment ID {segment_id}",
            enableTrace=True
        )
        
        # Extract the response
        messages = []
        for event in response.get('completion', {}).get('chunks', []):
            if 'chunk' in event:
                chunk = event['chunk']
                if 'message' in chunk:
                    content = chunk['message']['content']
                    if isinstance(content, list) and len(content) > 0:
                        text = content[0].get('text', '')
                        messages.append(text)
        
        return ''.join(messages)
    except Exception as e:
        st.error(f"Error invoking Bedrock agent: {str(e)}")
        return None

# Title
st.markdown('<p class="main-header">Wanderly Email Campaign Generator</p>', unsafe_allow_html=True)
st.markdown("Generate personalized marketing emails for flight promotions")
st.markdown("---")

# Create tabs for different steps
tab1, tab2, tab3 = st.tabs(["📋 Select Flight", "👥 User Segment", "✉️ Generate Email"])

with tab1:
    st.markdown('<p class="sub-header">1. Select Flight to Promote</p>', unsafe_allow_html=True)
    
    # Load flight data
    flight_df = read_s3_csv(BUCKET_NAME, 'data/travel_items.csv')
    
    if flight_df is not None:
        # Filter to only show promotions
        promo_flights = flight_df[flight_df['PROMOTION'] == 'Yes']
        
        if promo_flights.empty:
            st.warning("No promotional flights found.")
        else:
            # Display flights as a table
            st.markdown('<div class="info-box">Select one of the available promotional flights below:</div>', unsafe_allow_html=True)
            st.dataframe(
                promo_flights[['ITEM_ID', 'SRC_CITY', 'DST_CITY', 'AIRLINE', 'MONTH', 'DYNAMIC_PRICE']].reset_index(drop=True),
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
            
            # Select a flight
            selected_flight_id = st.selectbox(
                "Choose a flight to create a campaign for:",
                options=promo_flights['ITEM_ID'].tolist(),
                format_func=lambda x: f"{promo_flights[promo_flights['ITEM_ID']==x]['SRC_CITY'].iloc[0]} to {promo_flights[promo_flights['ITEM_ID']==x]['DST_CITY'].iloc[0]} - {promo_flights[promo_flights['ITEM_ID']==x]['AIRLINE'].iloc[0]}"
            )
            
            if selected_flight_id:
                flight_details = promo_flights[promo_flights['ITEM_ID'] == selected_flight_id].iloc[0]
                st.session_state['selected_flight_id'] = selected_flight_id
                st.session_state['flight_details'] = flight_details
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<div class="success-box">Flight details:</div>', unsafe_allow_html=True)
                    st.markdown(f"""
                    - **From:** {flight_details['SRC_CITY']}
                    - **To:** {flight_details['DST_CITY']}
                    - **Airline:** {flight_details['AIRLINE']}
                    - **Month:** {flight_details['MONTH']}
                    - **Duration:** {flight_details['DURATION_DAYS']} days
                    - **Price:** ${flight_details['DYNAMIC_PRICE']}
                    """)
                
                with col2:
                    st.markdown('<div class="info-box">Click below to continue to User Segment analysis</div>', unsafe_allow_html=True)
                    if st.button("Continue to User Segment →", use_container_width=True):
                        # Switch to the next tab
                        st.session_state['current_tab'] = 2
                        st.experimental_rerun()
    else:
        st.error("Could not load flight data.")

with tab2:
    st.markdown('<p class="sub-header">2. User Segment Details</p>', unsafe_allow_html=True)
    
    if 'selected_flight_id' not in st.session_state:
        st.info("Please select a flight first")
    else:
        selected_flight_id = st.session_state['selected_flight_id']
        flight_details = st.session_state['flight_details']
        
        st.markdown(f"""
        <div class="info-box">
        Analyzing user segment for flight: {flight_details['SRC_CITY']} to {flight_details['DST_CITY']} ({flight_details['AIRLINE']})
        </div>
        """, unsafe_allow_html=True)
        
        # Load segment data
        segments = read_s3_json(BUCKET_NAME, 'segments/batch_segment_input_ab3.json.out')
        
        if segments:
            # Find matching segment
            matching_segment = None
            for segment in segments:
                if segment.get('input', {}).get('itemId') == selected_flight_id:
                    matching_segment = segment
                    break
            
            if matching_segment:
                user_list = matching_segment.get('output', {}).get('usersList', [])
                st.session_state['user_list'] = user_list
                
                if user_list:
                    st.markdown(f'<div class="success-box">✅ Found {len(user_list)} users in this segment!</div>', unsafe_allow_html=True)
                    
                    # Load user data to show distribution
                    user_df = read_s3_csv(BUCKET_NAME, 'data/travel_users.csv')
                    if user_df is not None:
                        segment_users = user_df[user_df['USER_ID'].isin(user_list)]
                        tier_counts = segment_users['MEMBER_TIER'].value_counts()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Display as pie chart
                            st.subheader("User Tier Distribution")
                            st.bar_chart(tier_counts)
                        
                        with col2:
                            st.subheader("User Demographics")
                            st.markdown(f"""
                            - **Total users:** {len(user_list)}
                            - **Gold members:** {tier_counts.get('Gold', 0)}
                            - **Silver members:** {tier_counts.get('Silver', 0)}
                            - **Regular members:** {tier_counts.get('Member', 0)}
                            """)
                        
                        st.markdown('<div class="info-box">Click below to generate a personalized email campaign for this segment</div>', unsafe_allow_html=True)
                        if st.button("Generate Email Campaign →", use_container_width=True):
                            # Switch to the next tab
                            st.session_state['current_tab'] = 3
                            st.experimental_rerun()
                    else:
                        st.error("Could not load user data")
                else:
                    st.warning("No users found in this segment")
            else:
                # No exact match found, show all available segments
                st.warning(f"No exact segment match found for flight ID: {selected_flight_id}")
                
                # Display all available segments as an alternative
                segment_options = []
                for segment in segments:
                    item_id = segment.get('input', {}).get('itemId')
                    user_count = len(segment.get('output', {}).get('usersList', []))
                    segment_options.append({"item_id": item_id, "user_count": user_count})
                
                st.markdown("Available segments:")
                for option in segment_options:
                    if option["item_id"] in flight_df['ITEM_ID'].values:
                        flight_info = flight_df[flight_df['ITEM_ID'] == option["item_id"]].iloc[0]
                        st.markdown(f"- {flight_info['SRC_CITY']} to {flight_info['DST_CITY']} - {option['user_count']} users (ID: {option['item_id']})")
        else:
            st.error("No segment data available.")

with tab3:
    st.markdown('<p class="sub-header">3. Generate Email Campaign</p>', unsafe_allow_html=True)
    
    if 'selected_flight_id' not in st.session_state or 'user_list' not in st.session_state:
        st.info("Please complete the previous steps first")
    else:
        selected_flight_id = st.session_state['selected_flight_id']
        flight_details = st.session_state['flight_details']
        user_list = st.session_state['user_list']
        
        st.markdown(f"""
        <div class="info-box">
        Ready to generate email campaign for {flight_details['SRC_CITY']} to {flight_details['DST_CITY']} targeting {len(user_list)} users
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Generate Personalized Email", use_container_width=True):
            with st.spinner("Generating personalized email content..."):
                # Call the Bedrock Agent
                email_content = generate_email_with_agent(selected_flight_id)
                
                if email_content:
                    st.session_state['generated_email'] = email_content
                    
                    # Try to extract subject line using common patterns
                    subject_line = ""
                    if "Subject:" in email_content:
                        subject_line = email_content.split("Subject:")[1].split("\n")[0].strip()
                    elif "SUBJECT:" in email_content:
                        subject_line = email_content.split("SUBJECT:")[1].split("\n")[0].strip()
                    
                    st.markdown(f'<div class="success-box">✅ Email campaign generated successfully!</div>', unsafe_allow_html=True)
                    
                    # Show email preview
                    st.subheader("Email Preview")
                    
                    # Display in a nice format with tabs for different views
                    preview_tab1, preview_tab2 = st.tabs(["Formatted View", "Raw Text"])
                    
                    with preview_tab1:
                        # Try to extract and format the email nicely
                        if subject_line:
                            st.markdown(f"### {subject_line}")
                        
                        # Format the body - replace newlines with HTML breaks for better display
                        email_body = email_content
                        if subject_line:
                            # Remove the subject line from the body if we extracted it
                            email_body = email_body.replace(f"Subject: {subject_line}", "")
                            email_body = email_body.replace(f"SUBJECT: {subject_line}", "")
                        
                        # Convert markdown-like formatting to actual markdown
                        email_body = email_body.replace("**", "**").replace("__", "**")
                        
                        st.markdown(email_body)
                    
                    with preview_tab2:
                        # Show the raw text
                        st.text_area("Email Content", email_content, height=400)
                    
                    # Download button
                    st.download_button(
                        label="Download Email Content",
                        data=email_content,
                        file_name=f"wanderly_campaign_{flight_details['DST_CITY'].lower().replace(' ', '_')}.txt",
                        mime="text/plain"
                    )
                    
                    # User list download
                    st.subheader("Target User List")
                    st.markdown(f"This campaign will target **{len(user_list)}** users")
                    
                    # Convert user list to CSV for download
                    user_csv = "USER_ID\n" + "\n".join(user_list)
                    st.download_button(
                        label="Download User List (CSV)",
                        data=user_csv,
                        file_name=f"wanderly_users_{flight_details['DST_CITY'].lower().replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("Failed to generate email content. Please try again.")
        
        # Display previously generated email if available
        if 'generated_email' in st.session_state:
            st.subheader("Previously Generated Email")
            st.text_area("Email Content", st.session_state['generated_email'], height=300)

# Initialize tab state if needed
if 'current_tab' in st.session_state:
    # JavaScript to switch tabs
    tab_index = st.session_state['current_tab'] - 1  # 0-based index
    js = f"""
    <script>
        // Wait for DOM to load
        document.addEventListener('DOMContentLoaded', (event) => {{
            // Get the tab buttons
            const tabs = document.querySelectorAll('button[role="tab"]');
            if (tabs.length > {tab_index}) {{
                // Click the tab
                setTimeout(() => tabs[{tab_index}].click(), 100);
            }}
        }});
    </script>
    """
    st.markdown(js, unsafe_allow_html=True)
    # Clear the state after use
    del st.session_state['current_tab']