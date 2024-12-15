"""
This application annotates PDFs at a given URL or file path.

Requires the following packages
  pip install annotateai streamlit streamlit-pdf-viewer
"""

import os
import platform

import streamlit as st

from streamlit_pdf_viewer import pdf_viewer

from annotateai import Annotate


class Application:
    """
    Main application.
    """

    def __init__(self):
        """
        Creates a new application.
        """

        self.annotate = Annotate(
            os.environ.get(
                "LLM",
                (
                    "NeuML/Llama-3.1_OpenScholar-8B-AWQ"
                    if platform.machine() in ("x86_64", "AMD")
                    else "bartowski/Llama-3.1_OpenScholar-8B-GGUF/Llama-3.1_OpenScholar-8B-Q4_K_M.gguf"
                ),
            )
        )

    def run(self):
        """
        Main rendering logic.
        """

        # Example papers
        examples = {
            "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks": "https://arxiv.org/pdf/2005.11401",
            "HunyuanVideo: A Systematic Framework For Large Video Generative Models": "https://arxiv.org/pdf/2412.03603v2",
            "OpenDebateEvidence: A Massive-Scale Argument Mining and Summarization Dataset": "https://arxiv.org/pdf/2406.14657",
        }

        # UI components
        url, selected, download, render = st.empty(), st.empty(), st.empty(), st.empty()

        # Get examples
        selected = selected.pills("**or try these examples**", examples, key="example", on_change=self.onchange)
        selected = st.session_state.get("selected")

        # Create URL input using selected example, if applicable
        url = url.text_input("**URL or Local File Path**", value=examples.get(selected, ""))

        # Annotate the URL
        if url:
            # Build the annotation file for URL
            with st.spinner(f"Generating annotations for {url}"):
                output = self.build(url)

            # Get url file name
            _, name = os.path.split(url)
            name = name if name.lower().endswith(".pdf") else f"{name}.pdf"

            # Download file
            with open(output, "rb") as outfile:
                download.download_button(label=f"Download {name}", data=outfile, file_name=name, mime="application/pdf", type="primary")

            # File previews
            with render.container():
                # Limit file previews to files <= 20 MB due to data url size limitations
                if os.path.getsize(output) >= 20 * 1024 * 1024:
                    st.info("File too big to preview. Click download to view.")
                else:
                    st.divider()
                    pdf_viewer(output, render_text=True)

    def onchange(self):
        """
        Sets the selected example and clears the UI component.
        """

        st.session_state.selected = st.session_state.example
        st.session_state.example = None

    # pylint: disable=E0213
    @st.cache_data(show_spinner=False)
    def build(_self, url):
        """
        Builds an annotation for url.

        Args:
            url: url to annotate

        Returns:
            annotated file path
        """

        return _self.annotate(url)


@st.cache_resource(show_spinner="Initializing application...")
def create():
    """
    Creates and caches a Streamlit application.

    Returns:
        Application
    """

    return Application()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    st.set_page_config(
        page_title="Annotate papers using LLMs",
        page_icon="📝",
        layout="centered",
        initial_sidebar_state="auto",
        menu_items=None,
    )
    st.markdown("## 📝 Annotate papers using LLMs")

    st.markdown(
        """
This application automatically annotates a paper using LLMs.
 
_Try PDFs from [arXiv](https://arxiv.org/), [PubMed](https://pubmed.ncbi.nlm.nih.gov/),
[bioRxiv](https://www.biorxiv.org/) or [medRxiv](https://www.medrxiv.org/)!_
"""
    )

    # Create and run application
    app = create()
    app.run()
