import streamlit as st
import random
import streamlit.components.v1 as components

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title = "Retail Analytics Gamechanger",
    page_icon = "✨",
    layout = "centered",
)

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown("""
    <style>
        .main {
            padding: 2rem;
            background-color: #f9f9f9;
        }
        .stTextInput > div > div > input {
            border-radius: 10px;
        }
        .block-container {
            max-width: 800px;
            margin: auto;
        }
        .result-table {
            margin-top: 20px;
            border-radius: 10px;
            overflow: hidden;
        }
    </style>
""", unsafe_allow_html = True)

# Remove the header with the 'deploy' button
hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .css-18e3th9, .block-container {padding-top: 20px; padding-bottom: 0;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html = True)

# -----------------------------
# App Title
# -----------------------------
st.title("✨ Retail Analytics Gamechanger")

# -----------------------------
# Initialize session state
# -----------------------------
if "result_df" not in st.session_state:
    st.session_state.result_df = None  # to store the processed table

# -----------------------------
# Input Section
# -----------------------------
sample_placeholders = [
    'Show me total sales by product category for last month.', 'What were the top 5 best-selling products this quarter?', 'Show total revenue by store location for the current year.',
    'How have sales trended week over week this month?', 'Which day had the highest sales last week?', 'Compare total online vs. offline sales for the last 3 months.',
    'Show the monthly revenue growth rate for the past year.', 'What is the average order value by customer segment?', 'Which customers made more than 3 purchases last month?',
    'List the top 10 customers by total purchase value this year.', 'How many new customers did we acquire in Q2?', 'Show customer retention rate over the past 6 months.',
    'What’s the average purchase frequency per customer category?', 'Which products have the lowest stock levels right now?', 'Show products with no sales in the last 60 days.',
    'Which suppliers contribute the most to total product sales?', 'Display inventory value by product category.', 'Compare total sales between this year and last year.',
    'What are the peak sales hours during weekends?', 'Show daily sales for the last 30 days in a table format.'
]

user_input = st.text_area(
    "Enter your query:",
    placeholder = 'e.g., ' + random.choice(sample_placeholders),
    height = 150,
    key = "query_input"
)

# --- Hidden button (triggered by Enter) ---
run_button = st.button("Display Result", type="primary", key="run_button")

# -----------------------------
# Inject JavaScript
# -----------------------------
components.html(
    """
    <script>
    (function(){
        const textarea = window.parent.document.querySelector('textarea[aria-label="Enter your query:"]')
                       || window.parent.document.querySelector('textarea');
        const button = Array.from(window.parent.document.querySelectorAll('button'))
                          .find(b => b.innerText && b.innerText.trim().toLowerCase().includes('display result'));
        if (!textarea || !button) return;

        textarea.addEventListener('keydown', function(e) {
            // ENTER → act like Ctrl+Enter; Shift+Enter → newline
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();

                // Mimic Ctrl+Enter (Streamlit listens for this)
                const ctrlEnter = new KeyboardEvent('keydown', {key:'Enter', code:'Enter', ctrlKey:true, bubbles:true});
                textarea.dispatchEvent(ctrlEnter);

                // Visual feedback
                button.style.transition = 'all 0.12s ease';
                button.style.transform = 'scale(0.97)';
                button.style.opacity = '0.75';
                setTimeout(() => {
                    button.style.transform = '';
                    button.style.opacity = '';
                }, 120);

                // Trigger the button click shortly after (almost instant)
                setTimeout(() => button.click(), 30);
            }
        }, false);
    })();
    </script>
    """,
    height=0,
)

# Function 
@st.cache_resource(ttl = 7200, show_spinner = False)
def process_user_query(user_input: str):
    import retriever_executor
    return retriever_executor.get_data_from_query(user_input)

# -----------------------------
# Process Button
# -----------------------------
if run_button:
    st.session_state.button_pressed = True
    if user_input.strip():
        with st.spinner("Processing your query..."):
            
            print(f"User Query: {user_input}")

            # Function call to process the user query and get the resulting data table
            result_df = process_user_query(user_input)

            # Save to session state
            st.session_state.result_df = result_df
            
    else:
        st.warning("⚠️ Please enter some text before processing.")

# -----------------------------
# Display results if available
# -----------------------------
if type(st.session_state.result_df) == str:
    st.warning("⚠️ No data could be found for this request. Please refine your query and try again.")

elif st.session_state.result_df is not None:
    print("Query executed successfully.")
    st.success("✅ Processing complete!")
    st.markdown("### Results")
    st.dataframe(
        st.session_state.result_df.style.hide(axis = "index").set_table_styles(
            [{'selector': 'thead th', 'props': [('background-color', '#4B9CD3'),
                                                ('color', 'white'),
                                                ('font-weight', 'bold')]}]
        ),
        use_container_width = True, hide_index = True
    )

else:
    st.info("🪄 Enter a query and click **Display Results**.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Built with Streamlit by Abhay and Prem")
