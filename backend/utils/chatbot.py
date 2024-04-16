import time
from openai import OpenAI
import os
from langchain.vectorstores import Chroma
from typing import List, Tuple
import re
import ast
import html
from utils.load_config import LoadConfig

APPCFG = LoadConfig()


class ChatBot:
    @staticmethod
    def respond(chatbot: List, message: str, data_type: str = "Preprocessed doc", temperature: float = 0.0) -> Tuple:

        if data_type == "Preprocessed doc":
            # directories
            if os.path.exists(APPCFG.persist_directory):
                vectordb = Chroma(persist_directory=APPCFG.persist_directory,
                                  embedding_function=APPCFG.embedding_model)
            else:
                response="VectorDB does not exist. Please first execute the 'upload_data_manually.py' module."
                return response, None

        elif data_type == "Upload doc: Process for RAG":
            if os.path.exists(APPCFG.custom_persist_directory):
                vectordb = Chroma(persist_directory=APPCFG.custom_persist_directory,
                                  embedding_function=APPCFG.embedding_model)
            else:
                response="No file was uploaded. Please first upload your files using the 'upload' button."
                return response, None

        docs = vectordb.similarity_search(message, k=APPCFG.k)
        print(docs)
        question = "# User new question:\n" + message
        references = ChatBot.clean_references(docs)
        # Memory: previous two Q&A pairs
        chat_history = f"Chat history:\n {str(chatbot[-APPCFG.number_of_q_a_pairs:])}\n\n"
        prompt = f"{chat_history}{references}{question}"
        print("========================")
        print(prompt)
        client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=APPCFG.llm_engine,
            messages=[
                {"role": "system", "content": APPCFG.llm_system_role},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            # stream=False
        )
        
        res=response.choices[0].message.content
        print(res)
        time.sleep(2)

        return res, references

    @staticmethod
    def clean_references(documents: List) -> str:
        """
        Clean and format references from retrieved documents.

        Parameters:
            documents (List): List of retrieved documents.

        Returns:
            str: A string containing cleaned and formatted references.
        """
        documents = [str(x)+"\n\n" for x in documents]
        markdown_documents = ""
        counter = 1
        for doc in documents:
            # Extract content and metadata
            content, metadata = re.match(
                r"page_content=(.*?)( metadata=\{.*\})", doc).groups()
            metadata = metadata.split('=', 1)[1]
            metadata_dict = ast.literal_eval(metadata)

            # Decode newlines and other escape sequences
            content = bytes(content, "utf-8").decode("unicode_escape")

            # Replace escaped newlines with actual newlines
            content = re.sub(r'\\n', '\n', content)
            # Remove special tokens
            content = re.sub(r'\s*<EOS>\s*<pad>\s*', ' ', content)
            # Remove any remaining multiple spaces
            content = re.sub(r'\s+', ' ', content).strip()

            # Decode HTML entities
            content = html.unescape(content)

            # Replace incorrect unicode characters with correct ones
            content = content.encode('latin1').decode('utf-8', 'ignore')

            # Remove or replace special characters and mathematical symbols
            # This step may need to be customized based on the specific symbols in your documents
            content = re.sub(r'â', '-', content)
            content = re.sub(r'â', '∈', content)
            content = re.sub(r'Ã', '×', content)
            content = re.sub(r'ï¬', 'fi', content)
            content = re.sub(r'â', '∈', content)
            content = re.sub(r'Â·', '·', content)
            content = re.sub(r'ï¬', 'fl', content)

            # Append cleaned content to the markdown string with two newlines between documents
            markdown_documents += f"# Retrieved content {counter}:\n" + content + "\n\n"
            counter += 1

        return markdown_documents
