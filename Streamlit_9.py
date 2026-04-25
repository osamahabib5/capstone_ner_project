import streamlit as st
from groq import Groq, APIStatusError, RateLimitError
import time

# Initialize the Groq client
# Ensure GROQ_API_KEY is set in your Streamlit Secrets or Environment Variables
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def estimate_token_count(text):
    """
    A rough estimation: 1 token is approx 4 characters or 0.75 words.
    The limit is 8K/minute, so we want to stay well under that per request.
    """
    return len(text) // 4

def groq_stream_generator(full_query):
    model_name = "openai/gpt-oss-120b" # Updated based on your image
    
    # 1. Check token safety before sending
    estimated_tokens = estimate_token_count(full_query)
    if estimated_tokens > 7000: # Leaving a 1K buffer for the response
        st.warning(f"⚠️ Query is large (~{estimated_tokens} tokens). This may hit your 8K/min limit.")

    # 2. Implementation of retry logic for Rate Limits
    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": full_query}
                ],
                stream=True,
            )

            for chunk in completion:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            
            # If successful, break the retry loop
            break

        except RateLimitError as e:
            if attempt < max_retries - 1:
                st.info(f"Rate limit reached. Retrying in {retry_delay}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                continue
            else:
                st.error("❌ Rate limit exceeded. Please wait a minute before trying again.")
                yield "Error: Rate limit exceeded. Please try again in 60 seconds."
        
        except APIStatusError as e:
            # Handle other API issues (Auth, Model not found, etc.)
            st.error(f"Groq API Error: {e.status_code} - {e.message}")
            break
        
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            break

# --- Streamlit UI Logic ---
st.title("AI & Database Assistant")

user_input = st.text_area("Enter your query:", placeholder="Ask about historical records...")

if st.button("Generate"):
    if user_input:
        # Assuming 'full_query' is built here (e.g., combining context from DB + user input)
        # For this example, we'll use user_input as full_query
        full_query = user_input 
        
        with st.chat_message("assistant"):
            try:
                # This matches line 190 in your traceback
                st.write_stream(groq_stream_generator(full_query))
            except Exception as e:
                st.error(f"Stream Error: {e}")
    else:
        st.warning("Please enter a prompt.")
