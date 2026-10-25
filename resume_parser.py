import fitz


def extract_resume_text(uploaded_file):
    """Extract readable text from an uploaded PDF resume."""

    if uploaded_file is None:
        return ""

    try:
        pdf_bytes = uploaded_file.getvalue()

        if not pdf_bytes:
            return ""

        text_parts = []

        with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
            for page in pdf:
                page_text = page.get_text()
                if page_text:
                    text_parts.append(page_text)

        return "\n".join(text_parts).strip()

    except Exception as error:
        print(f"Resume extraction error: {error}")
        return ""