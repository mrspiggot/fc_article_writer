"""
This module provides a set of classes for processing documents, building vector stores,
and generating articles using AI models within a Streamlit UI.
"""

import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import BingSearchAPIWrapper
import time
import pprint
import tempfile
import ast
from datetime import datetime
from dateutil.relativedelta import relativedelta

from dotenv import load_dotenv
import os

class LuciDocumentProcessor:
    """
    A class to process document files, specifically for extracting text from PDF files.
    """
    def __init__(self, file_data):
        """
        Initializes the LuciDocumentProcessor with the file data.

        :param file_data: The binary data of the file to be processed.
        """
        self.file_data = file_data

    def get_text(self):
        """
        Extracts and returns the text from the loaded PDF file.

        :return: A string containing all the text extracted from the PDF file.
        """
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(self.file_data.getvalue())
            loader = PyPDFLoader(os.path.abspath(temp_file.name))
            pages = loader.load_and_split()
            return "".join(t.page_content for t in pages)

class LuciFactFinder:
    """
    Handles fact-finding of facts that need to be validated for AI generated articles.
    Improved version with better error handling, search query construction, and results display.
    """
    def __init__(self, ui_controller, facts_to_check):
        """
        Initializes the LuciFactFinder with a reference to the UI controller and the facts to check.
        """
        self.ui = ui_controller
        self.facts_to_check = facts_to_check
        self.search = BingSearchAPIWrapper()

    def find_facts(self):
        """
        Validates the facts by searching for them using a modified search query.
        """
        for fact in self.facts_to_check:
            search_results = self.perform_search_with_retries(fact)
            self.display_search_results(fact, search_results)

    def perform_search_with_retries(self, fact, retries=2, delay=2):
        """
        Attempts to perform the search with retries on failure.
        """
        for attempt in range(retries + 1):
            try:
                return self.execute_search(fact)
            except Exception as e:  # Consider catching more specific exceptions
                if attempt < retries:
                    time.sleep(delay)
                else:
                    raise e

    def execute_search(self, fact):
        """
        Executes the search query with modifications based on the UI settings.
        """
        prefix, suffix = self.get_search_query_modifiers()
        return self.search.results(f"{prefix}{fact}{suffix}", 3)

    def get_search_query_modifiers(self):
        """
        Constructs the prefix and suffix for the search query based on UI options.
        """
        if self.ui.latest_news_option == "Yes":
            prefix = "Get the latest answer for this question: "
            suffix = " after: " + (datetime.now() - relativedelta(months=2)).strftime('%Y-%m-%d')
        else:
            prefix = ""
            suffix = ""
        return prefix, suffix

    def display_search_results(self, fact, search_results):
        """
        Displays the search results in the Streamlit UI.
        """
        with st.expander(f"Fact: {fact}"):
            for result in search_results:
                st.write(f"Title: {result['title']}")
                st.write(f"Link: {result['link']}")
                st.write("\n")




class LuciFactChecker:
    """
    Handles fact-checking of generated articles using an AI model.
    """
    def __init__(self, ui_controller):
        """
        Initializes the LuciFactChecker with a reference to the UI controller.

        :param ui_controller: An instance of LuciUIController for UI interactions.
        """
        self.ui = ui_controller

    def fact_check(self, article):
        """
        Performs fact-checking on the given article.

        :param article: The article text to fact check.
        :return: A list of facts to check.
        """

        article_retriever = LuciVectorStoreManager(article).build_vectorstore()
        fc_model = ChatOpenAI(model=self.ui.selected_model)
        fc_prompt = ChatPromptTemplate.from_template('''Given this article {article} return a python tuple containing five strings. Each string is a fact to check from the article. Each string should be phrased as a direct question to pose to a fact-checking AI agent.''')
        fc_chain = ({"article": article_retriever} | fc_prompt | fc_model | StrOutputParser()).invoke("Please fact-check this article")
        # st.write(fc_chain)
        result_chain = ast.literal_eval(fc_chain)
        n=1

        for question in result_chain:
            st.write(f"{n}. {question}")
            n+=1


        return result_chain


class LuciVectorStoreManager:
    """
    Manages the creation and retrieval of vector stores for text data.
    """
    def __init__(self, text):
        """
        Initializes the LuciVectorStoreManager with text to be vectorized.

        :param text: The text to build the vector store from.
        """
        self.text = text

    def build_vectorstore(self):
        """
        Builds and returns a vector store from the provided text.

        :return: A retriever object for the constructed vector store.
        """
        vectorstore = FAISS.from_texts([self.text], embedding=OpenAIEmbeddings())
        return vectorstore.as_retriever()

class LuciUIController:
    """
    Handles the user interface and interaction in the Streamlit app.
    """
    def __init__(self):
        """
        Initializes the LuciUIController, setting up the UI components.
        """
        self.sidebar = st.sidebar
        self.model_options = ["gpt-4", "gpt-3.5-turbo-0613", "Gemini", "Luci-FT-Gen", "Claude2"]
        self.latest_news_option = None
        self.setup_ui()

    def setup_ui(self):
        """
        Sets up the user interface elements in the Streamlit sidebar and main area.
        """
        logo = "color_lucidate.png"
        st.sidebar.image(logo, width=120)
        self.sidebar.header("AI Settings")
        self.selected_model = self.sidebar.selectbox("Select AI Model", self.model_options)
        self.temperature = self.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
        self.latest_news_option = self.sidebar.radio(
            "Latest news for fact checking:",
            ("Yes", "No"),
            index=0  # Default to "Yes"
        )
        self.sidebar.header("Upload Files")
        self.sample_article = self.sidebar.file_uploader("Sample Article", type="pdf")
        self.essays = self.sidebar.file_uploader("Essays", type=["pdf"], accept_multiple_files=True)
        st.header("Article Generator:")
        self.prompt_text = st.text_area("Edit Prompt", """Write an article using the style of this document {style}. Replicate its approach to generating titles and subheading. Ensure that the subheadings relate to the content that follows to be as helpful as possible to the reader.  Think carefully about the principles of a good headline and apply these principles to make the headline as relevant, catchy and compelling as you can to encourage readership. Put a newline after each title and subheading. For the article base it only on the following content: {context}. Use only the content do not publish the names of the people responsible for the research.""", height=200)

class LuciArticleGenerator:
    """
    Generates articles using the provided AI model and user inputs.
    """
    def __init__(self, ui_controller):
        """
        Initializes the LuciArticleGenerator with a reference to the UI controller.

        :param ui_controller: An instance of LuciUIController for UI interactions.
        """
        self.ui = ui_controller

    def generate(self):
        """
        Generates an article based on the user inputs and outputs the result to the Streamlit interface.
        """
        # If a sample article is uploaded, process it to extract the text.
        # Otherwise, set style_text to an empty string.
        style_text = LuciDocumentProcessor(self.ui.sample_article).get_text() if self.ui.sample_article else ""

        # If one or more essays are uploaded, process each to extract their text.
        # Combine the text of all essays into one string, or set source_texts to an empty list if no essays are uploaded.
        source_texts = [LuciDocumentProcessor(essay).get_text() for essay in self.ui.essays] if self.ui.essays else []

        # Use the combined text from the essays to build a vector store for content retrieval.
        source_retriever = LuciVectorStoreManager(" ".join(source_texts)).build_vectorstore()

        # Use the text from the sample article to build a vector store for style retrieval.
        style_retriever = LuciVectorStoreManager(style_text).build_vectorstore()

        # Create a prompt template from the UI's prompt text.
        prompt = ChatPromptTemplate.from_template(self.ui.prompt_text)

        # Initialize the ChatOpenAI model using the selected AI model from the UI.
        model = ChatOpenAI(model=self.ui.selected_model)

        # Build a chain of operations using LangChain Expression Language (LCEL).
        # This chain combines the context from source_retriever, the style from style_retriever,
        # applies the prompt template, invokes the selected AI model, and parses the output to a string.
        result_chain = ({"context": source_retriever, "style": style_retriever} | prompt | model | StrOutputParser()).invoke("Please write an article in this style")
        # Output the result to the Streamlit interface.
        st.write(result_chain)


        return result_chain




if __name__ == "__main__":
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    BING_SUBSCRIPTION_KEY = os.getenv("BING_SUBSCRIPTION_KEY")
    BING_SEARCH_URL = os.getenv("BING_SEARCH_URL")

    ui_controller = LuciUIController()
    article_generator = LuciArticleGenerator(ui_controller)
    fact_checker = LuciFactChecker(ui_controller)

    if st.button("Run"):
        article = LuciArticleGenerator(ui_controller).generate()
        with st.spinner("To fact check the document the following questions need to be answered:"):
            facts_to_check = fact_checker.fact_check(article)
        with st.spinner("Fact checking in progress..."):
            fact_finder = LuciFactFinder(ui_controller, facts_to_check)
            fact_finder.find_facts()

