
import streamlit as st


def text_translator():

    st.header("📝 ISL Text Translator")

    st.write(
        "Convert recognized ISL alphabet letters "
        "into words and sentences."
    )

    # Initialize translated text
    if "translated_text" not in st.session_state:
        st.session_state.translated_text = ""

    # Enter text
    letter = st.text_input(
        "Enter recognized ISL letters or words:",
        placeholder="Example: HELLO"
    )

    col1, col2 = st.columns(2)

    # Add text automatically with spacing
    with col1:

        if st.button("Add Text"):

            new_text = letter.strip().upper()

            if new_text:

                if st.session_state.translated_text:
                    st.session_state.translated_text += " "

                st.session_state.translated_text += new_text

                st.rerun()

    # Add space manually
    with col2:

        if st.button("Add Space"):

            if st.session_state.translated_text:
                st.session_state.translated_text += " "

                st.rerun()

    st.divider()

    # Display translated sentence
    st.subheader("Translated Text")

    st.text_area(
        "Your sentence:",
        value=st.session_state.translated_text,
        height=120,
        disabled=True
    )

    # Character count
    st.metric(
        "Character Count",
        len(st.session_state.translated_text)
    )

    # Clear text
    if st.button("Clear Text"):

        st.session_state.translated_text = ""

        st.rerun()
