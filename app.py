import streamlit as st
import base64
import json
from functions import *
from bson.objectid import ObjectId

# Initialize session state variables
if "generate_clicked" not in st.session_state:
    st.session_state.generate_clicked = False
if "edited_lesson_plan" not in st.session_state:
    st.session_state.edited_lesson_plan = ""
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = False
if "lesson_plan" not in st.session_state:
    st.session_state.lesson_plan = ""
if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = {}
if "document_id" not in st.session_state:
    st.session_state.document_id = None

# Function to reset inputs
def reset_inputs():
    """Resets all session state variables and refreshes the page."""
    st.session_state.clear()
    st.rerun()

# Function to display PDF
def display_pdf(uploaded_file):
    """Display a PDF file uploaded to Streamlit."""
    bytes_data = uploaded_file.getvalue()
    base64_pdf = base64.b64encode(bytes_data).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Function to load Streamlit sidebar
def load_streamlit_page():
    """Load the Streamlit sidebar for inputs."""
    st.set_page_config(page_title="LLM Tool", layout="wide")

    # Sidebar for Inputs
    st.sidebar.markdown(
        "<h2 style='color: var(--text-color, #1E90FF);'>📄 Curriculum PDF Analyzer</h2>",
        unsafe_allow_html=True
    )
    st.sidebar.markdown(
        """
        🔹 **Instructions:**  
        1. Upload a **PDF curriculum document**.  
        2. Enter the **class/grade** (e.g., 'Grade 10').  
        3. Specify the **topic** (e.g., 'Mathematics').  
        4. Select the **number of days** for the lesson plan.  
        5. Click **Generate** to process.  
        6. Use **Reset** to start over.
        """, unsafe_allow_html=True
    )

    # Upload PDF with icon
    uploaded_file = st.sidebar.file_uploader(
        "📤 Upload PDF", 
        type="pdf", 
        help="Upload a PDF curriculum document to analyze."
    )

    # Text Inputs with icons
    class_data = st.sidebar.text_input(
        "🏫 Class/Grade", 
        placeholder="e.g., Grade 10", 
        help="Enter the class or grade level."
    )
    topic = st.sidebar.text_input(
        "📚 Topic", 
        placeholder="e.g., Mathematics", 
        help="Specify the subject or topic."
    )
    days = st.sidebar.number_input(
        "📅 Number of Days", 
        min_value=1, 
        max_value=100, 
        value=1, 
        help="Set the duration of the lesson plan (each day = 50 mins)."
    )

    # Sidebar Buttons with styling
    col1, col2 = st.sidebar.columns(2)
    with col1:
        generate_button = st.button(
            "🚀 Generate", 
            key="generate", 
            help="Click to process the PDF and generate a lesson plan.",
            use_container_width=True
        )
    with col2:
        reset_button = st.button(
            "🔄 Reset", 
            key="reset", 
            on_click=reset_inputs, 
            help="Click to clear all inputs and start over.",
            use_container_width=True
        )

    return uploaded_file, class_data, topic, days, generate_button

# Load Sidebar Inputs
uploaded_file, class_data, topic, days, generate_button = load_streamlit_page()

# Handle Generate Button Click
if generate_button:
    st.session_state.generate_clicked = True
    with st.spinner("🔄 Processing PDF and generating lesson plan... Please wait!"):
        try:
            full_text, documents = extract_pdf_data(uploaded_file)
            analysis = analyze_text(full_text, class_data)
            vector_store = create_vectorstore_from_texts(documents, openai_api_key, uploaded_file.name)

            data = json.loads(analysis)
            data["duration"] = f"{days} days"
            document_id = push_to_mongo(data)
            st.session_state.document_id = document_id
            st.success("✅ Extracted data stored in MongoDB with updated duration!", icon="✅")

            st.session_state.extracted_data = data

            lesson_plan = generate_lesson_plan(client, class_data, days, vector_store, topic)
            st.session_state.lesson_plan = lesson_plan
            st.session_state.edited_lesson_plan = lesson_plan

            update_lesson_plan_in_mongo(document_id, lesson_plan)
            st.success("✅ Lesson plan saved to MongoDB!", icon="✅")

        except Exception as e:
            st.error(f"❌ Error processing PDF: {str(e)}", icon="❌")

# Main Content Section
st.markdown(
    "<h1 style='text-align: center; color: var(--text-color, #1E90FF);'>📘 Open Pedagogy - AI Parsing</h1>",
    unsafe_allow_html=True
)
st.divider()

if uploaded_file is not None and class_data and topic and days and st.session_state.generate_clicked:
    # Tabs for output display
    tab_plan, tab_info = st.tabs(["📚 Lesson Plan", "📊 Extracted Info"])

    with tab_plan:
        st.markdown(
            "<h3 style='color: var(--text-color, #4682B4);'>📚 Lesson Plan</h3>",
            unsafe_allow_html=True
        )

        # Edit Mode Handling without custom theme
        if st.session_state.edit_mode:
            edited_lesson_plan = st.text_area(
                "Edit the lesson plan below:",
                value=st.session_state.edited_lesson_plan,
                height=400,
                help="Modify the lesson plan as needed."
            )
            if st.button("💾 Save Changes", key="save_changes", help="Save your edits to the lesson plan."):
                st.session_state.edited_lesson_plan = edited_lesson_plan
                st.success("✅ Changes saved successfully!", icon="✅")
                st.session_state.edit_mode = False

                if "document_id" in st.session_state:
                    update_lesson_plan_in_mongo(st.session_state.document_id, edited_lesson_plan)
                    st.success("✅ Updated lesson plan saved to MongoDB!", icon="✅")
                else:
                    st.error("❌ Document ID not found. Cannot save to MongoDB.", icon="❌")
                st.rerun()
        else:
            # Display in non-edit mode with theme-aware styling
            st.markdown(
                f"""
                <div style='background-color: #F0F8FF; color: #333333; padding: 15px; border-radius: 10px;'>
                    {st.session_state.edited_lesson_plan}
                </div>
                """,
                unsafe_allow_html=True
            )

        # Buttons for interaction
        col1, col2 = st.columns(2)
        with col1:
            st.button(
                "✏️ Toggle Edit Mode",
                key="toggle_edit",
                on_click=lambda: st.session_state.__setitem__("edit_mode", not st.session_state.edit_mode),
                help="Switch between viewing and editing the lesson plan."
            )
        with col2:
            st.download_button(
                label="⬇️ Download Lesson Plan",
                data=st.session_state.edited_lesson_plan,
                file_name=f"lesson_plan_{class_data.lower().replace(' ', '_')}.md",
                mime="text/markdown",
                help="Download the lesson plan as a Markdown file."
            )

    with tab_info:
        st.markdown(
            "<h3 style='color: var(--text-color, #4682B4);'>📊 Extracted Information Summary</h3>",
            unsafe_allow_html=True
        )
        with st.expander("View Extracted Data", expanded=True):
            st.json(st.session_state.extracted_data, expanded=False)
else:
    st.warning("⚠️ Please upload a PDF, enter class details, topic, days, and click 'Generate'.", icon="⚠️")