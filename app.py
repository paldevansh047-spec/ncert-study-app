# app.py - NCERT Study App with Google Gemini API
import os
import PyPDF2
import gradio as gr
import google.generativeai as genai
import re

# === GET API KEY FROM ENVIRONMENT ===
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("⚠️ WARNING: GEMINI_API_KEY not set. App will not work.")

# === INITIALIZE GEMINI CLIENT ===
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# === PDF PROCESSING FUNCTIONS ===
def extract_text_from_pdf(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def chunk_text(text, chunk_size=1500):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def find_relevant_chunks(question, chunks, top_k=3):
    question_words = set(question.lower().split())
    scored_chunks = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(question_words.intersection(chunk_words))
        scored_chunks.append((score, chunk))
    scored_chunks.sort(reverse=True)
    return [chunk for score, chunk in scored_chunks[:top_k] if score > 0]

def generate_questions(topic, num_questions=5, grade_level="class 10"):
    if not GEMINI_API_KEY:
        return "❌ API key not configured."
    try:
        prompt = f"""
        You are an expert NCERT teacher for {grade_level} students.
        Topic: {topic}
        Generate {num_questions} practice questions.
        Format: Q1. [Question] Q2. [Question] ...
        Make questions: mix of easy, medium, hard, based on NCERT syllabus.
        """
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error: {str(e)}"

def answer_question(question, context_text=""):
    if not GEMINI_API_KEY:
        return "❌ API key not configured."
    try:
        if context_text:
            prompt = f"""
            You are a helpful NCERT tutor.
            Textbook content: {context_text[:8000]}
            Student question: {question}
            Answer based on the textbook. If not found, say so.
            """
        else:
            prompt = f"""
            You are a helpful NCERT tutor.
            Student question: {question}
            Answer in a simple, clear way.
            """
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error: {str(e)}"

def process_query(question, pdf_file, topic, num_questions, grade_level, mode):
    context_text = ""
    context_preview = "📚 No PDF uploaded. Using general knowledge."
    if pdf_file is not None:
        text = extract_text_from_pdf(pdf_file)
        if "Error" not in text:
            chunks = chunk_text(text)
            relevant = find_relevant_chunks(question, chunks)
            if relevant:
                context_text = "\n".join(relevant)
                context_preview = f"📚 Found {len(relevant)} sections from PDF"
            else:
                context_preview = "📚 No relevant content found."
        else:
            context_preview = text
    if mode == "Generate Questions":
        return generate_questions(topic, int(num_questions), grade_level), context_preview, ""
    else:
        answer = answer_question(question, context_text)
        return "", context_preview, answer

# === GRADIO INTERFACE ===
with gr.Blocks(theme=gr.themes.Soft(), title="NCERT Study App") as demo:
    gr.Markdown("# 📚 NCERT Study Assistant\n**Ask questions or generate practice questions!**")
    with gr.Row():
        with gr.Column(scale=1):
            mode = gr.Radio(["Ask Question", "Generate Questions"], label="Select Mode", value="Ask Question")
            pdf_file = gr.File(label="📄 Upload NCERT PDF (Optional)", file_types=[".pdf"])
            question_input = gr.Textbox(label="❓ Ask Your Question", placeholder="Type your question...", lines=3)
            topic_input = gr.Textbox(label="📝 Topic for Questions", placeholder="e.g., Photosynthesis", lines=1)
            num_questions = gr.Slider(3, 10, value=5, step=1, label="Number of questions")
            grade_level = gr.Dropdown(["class 6", "class 7", "class 8", "class 9", "class 10"], label="Grade Level", value="class 10")
            submit_btn = gr.Button("🚀 Go!", variant="primary")
        with gr.Column(scale=1):
            context_output = gr.Textbox(label="📖 Found in Textbook", lines=4, interactive=False)
            questions_output = gr.Textbox(label="📝 Generated Questions", lines=8, interactive=False)
            answer_output = gr.Textbox(label="💡 Answer", lines=8, interactive=False)
    submit_btn.click(process_query, inputs=[question_input, pdf_file, topic_input, num_questions, grade_level, mode], outputs=[questions_output, context_output, answer_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=10000)
