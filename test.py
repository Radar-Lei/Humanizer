from datasets import load_dataset
import os
import subprocess
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Get the default gateway IP dynamically
try:
    hostip = subprocess.check_output("ip route | grep default | awk '{print $3}'", shell=True).decode().strip()
    port = 7890
    PROXY_HTTP = f"http://{hostip}:{port}"
    
    # Set proxy environment variables
    os.environ['http_proxy'] = PROXY_HTTP
    os.environ['https_proxy'] = PROXY_HTTP
    print(f"Using proxy: {PROXY_HTTP}")
except Exception as e:
    print(f"Could not set proxy: {e}")
# Load the dataset
df = load_dataset("neuralwork/arxiver")

# Convert to pandas DataFrame for easier viewing
df = df['train'].to_pandas()

# from langchain.document_loaders import CSVLoader
separators = ["\n\n", "\n", " ", "", "."]

# Get your splitter ready
text_splitter = RecursiveCharacterTextSplitter(separators=separators, chunk_size=1000, chunk_overlap=5)

# Split your docs into texts
df = text_splitter.create_documents(df['markdown'].values)


# Embeddings
hg_embeddings = HuggingFaceEmbeddings()

from langchain_community.vectorstores import Chroma
persist_directory = 'docs/chroma_rag/'
economic_langchain_chroma = Chroma.from_documents(
    documents=df,
    collection_name="humanizer_data",
    embedding=hg_embeddings,
    persist_directory=persist_directory
)
