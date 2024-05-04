import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import YoutubeLoader
import getpass
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from pytube import YouTube

load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY not found in environment variables or .env file.")

def get_video_id(url):
    try:
        yt = YouTube(url)
        return yt.video_id
    except Exception as e:
        print(f"Error: {e}")
        return None


def chunk_data(document):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, chunk_overlap=50)
    doc = text_splitter.split_documents(document)
    return doc
def format_results(results):
    unique_content = set()
    formatted_results = []

    for result in results:
        content = result.page_content
        if content not in unique_content:
            unique_content.add(content)
            formatted_results.append(content)

    return "\n".join(formatted_results)

def main():
    st.set_page_config(page_title="YouTube Video Search")
    st.title("YouTube Video Search")

    # Get YouTube URL input from the user
    youtube_url = st.text_input("Enter YouTube URL", "https://www.youtube.com/watch?v=erUfLIi9OFM")
    video_id = get_video_id(youtube_url)
    if video_id:
        loader = YoutubeLoader.from_youtube_url(f"https://www.youtube.com/watch?v={video_id}", add_video_info=False)
        document = loader.load()
        # ...
    else:
        st.error("Invalid YouTube URL")
    # Load the video
    loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=False)
    document = loader.load()

    # Chunk the data
    chunked = chunk_data(document=document)

    # Set up embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    os.environ['PINECONE_API_KEY'] = "34b731fe-378d-4afd-b668-ec12e07435db"
    index_name = "ytvideoresource"
    vectorstore = PineconeVectorStore.from_documents(chunked, index_name=index_name, embedding=embeddings)

    # Get user query
    query = st.text_input("Enter your query")

    # Perform similarity search
    if query:
        results = vectorstore.similarity_search(query)
        formatted_results = format_results(results)
        st.write(formatted_results)

if __name__ == "__main__":
    main()