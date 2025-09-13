import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModel, pipeline
import torch
import soundfile as sf
import io
import numpy as np


# --- Page Configuration ---
st.set_page_config(
    page_title="EchoVerse: AI Text-to-Speech",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
.main-header {
    font-size: 3.5rem;
    font-weight: 700;
    color: #4CAF50;
    text-align: center;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 0.5em;
}

.subheader {
    font-size: 1.25rem;
    color: #555;
    text-align: center;
    margin-bottom: 2em;
}

.stButton>button {
    width: 100%;
    border-radius: 20px;
    background-color: #4CAF50;
    color: white;
    font-size: 1.2rem;
    padding: 10px 20px;
    border: none;
    transition: all 0.2s ease-in-out;
}
.stButton>button:hover {
    background-color: #45a049;
    transform: translateY(-2px);
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.stTextArea, .stFileUploader {
    border-radius: 10px;
    border: 2px solid #ddd;
}

.text-container {
    padding: 20px;
    border: 1px solid #eee;
    border-radius: 10px;
    background-color: black;
}
</style>
""", unsafe_allow_html=True)


# --- 1. Load the Hugging Face models ---
@st.cache_resource
def load_llm_model():
    """Loads the IBM Granite model for text rewriting."""
    try:
        tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.3-2b-instruct")
        model = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-3.3-2b-instruct")
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load LLM model: {e}")
        return None, None

@st.cache_resource
def load_tts_model():
    """Loads the Suno Bark TTS model."""
    try:
        processor = AutoProcessor.from_pretrained("suno/bark")
        model = AutoModel.from_pretrained("suno/bark")
        return processor, model
    except Exception as e:
        st.error(f"Failed to load TTS model: {e}")
        return None, None

# Assign models
tokenizer, llm_model = load_llm_model()
tts_processor, tts_model = load_tts_model()

# --- 2. Define the core functions ---

def rewrite_text_with_llm(text, tone):
    """Rewrites the input text into a specified tone using the LLM."""
    if not llm_model:
        return "LLM model not loaded."
    
    prompt_template = f"Rewrite the following text in a {tone} tone. The text is: '{text}'"
    messages = [{"role": "user", "content": prompt_template}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    outputs = llm_model.generate(**inputs, max_new_tokens=100)
    rewritten_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return rewritten_text.strip()

def get_audio_from_text(text, voice_preset):
    """Converts the rewritten text into audio using the Bark TTS model."""
    if not tts_processor or not tts_model:
        return None

    inputs = tts_processor(text, voice_preset=voice_preset, return_tensors="pt")
    
    # Generate audio
    speech_output = tts_model.generate(**inputs, do_sample=True, fine_temperature=0.4, coarse_temperature=0.8)
    
    # Get the audio waveform and sampling rate
    audio_waveform = speech_output[0].cpu().numpy()
    sampling_rate = tts_model.generation_config.sample_rate
    
    # Save the audio data to a buffer in a compatible format (e.g., WAV)
    buffer = io.BytesIO()
    sf.write(buffer, audio_waveform, sampling_rate, format='WAV')
    buffer.seek(0)
    return buffer.getvalue()

# --- 3. Create the Streamlit UI ---

def main():
    """The main Streamlit application function."""
    st.markdown("<h1 class='main-header'>EchoVerse 🎙️</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subheader'>Your text, transformed and brought to life. Perfect for a hackathon!</p>", unsafe_allow_html=True)

    # Initialize session state variables
    if 'original_text' not in st.session_state:
        st.session_state.original_text = ""
    if 'rewritten_text' not in st.session_state:
        st.session_state.rewritten_text = ""
    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = None
    if 'processing_done' not in st.session_state:
        st.session_state.processing_done = False
    
    # Input Method Selection
    st.subheader("1. Enter Your Text or Upload a File")
    input_method = st.radio("Choose your input method:", ("Enter Text Manually", "Upload .txt File"))

    input_text = ""
    if input_method == "Enter Text Manually":
        input_text = st.text_area("Type your text here:", height=150)
    else:
        uploaded_file = st.file_uploader("Choose a .txt file", type="txt")
        if uploaded_file is not None:
            input_text = uploaded_file.getvalue().decode("utf-8")
            st.text_area("File content:", input_text, height=150)
    
    st.subheader("2. Choose Voice & Tone")
    
    # Add voice selection for Bark
    voice_presets = {
        "English Speaker 1 (Male)": "v2/en_speaker_0",
        "English Speaker 2 (Female)": "v2/en_speaker_1",
        "English Speaker 3 (Male)": "v2/en_speaker_2",
        "English Speaker 4 (Female)": "v2/en_speaker_3",
        "English Speaker 5 (Male)": "v2/en_speaker_4",
        "English Speaker 6 (Female)": "v2/en_speaker_5",
        "English Speaker 7 (Male)": "v2/en_speaker_6",
        "English Speaker 8 (Female)": "v2/en_speaker_7",
        "English Speaker 9 (Male)": "v2/en_speaker_8",
    }
    selected_voice_name = st.selectbox("Select a Voice:", list(voice_presets.keys()))
    selected_voice_preset = voice_presets[selected_voice_name]
    
    tone_option = st.selectbox("Select a tone to rewrite your text:", ("Neutral", "Suspenseful", "Inspiring"))

    if st.button("Generate Audio"):
        if input_text:
            with st.spinner("Processing... This may take a moment."):
                # Store the original text in session state
                st.session_state.original_text = input_text
                
                # Rewrite text
                rewritten_text = rewrite_text_with_llm(st.session_state.original_text, tone_option)
                st.session_state.rewritten_text = rewritten_text

                # Generate audio with the selected voice
                audio_data = get_audio_from_text(st.session_state.rewritten_text, selected_voice_preset)
                st.session_state.audio_data = audio_data
                st.session_state.processing_done = True
        else:
            st.warning("Please enter some text or upload a file.")

    # Display the results only after processing is done
    if st.session_state.processing_done:
        st.success("Audio generated!")

        # Side-by-side text comparison
        st.subheader("3. Transformation")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h3>Original Text</h3>", unsafe_allow_html=True)
            st.markdown(f"<div class='text-container'>{st.session_state.original_text}</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<h3>Tone-Adapted Text</h3>", unsafe_allow_html=True)
            st.markdown(f"<div class='text-container'>{st.session_state.rewritten_text}</div>", unsafe_allow_html=True)

        # Audio output and download
        st.subheader("4. Listen & Download")
        if st.session_state.audio_data:
            st.audio(st.session_state.audio_data, format="audio/wav")

            st.download_button(
                label="Download Audio as WAV",
                data=st.session_state.audio_data,
                file_name="echoverse_output.wav",
                mime="audio/wav"
            )
        else:
            st.error("Could not generate audio. Please check the model loading status.")

if __name__ == "__main__":
    main()
