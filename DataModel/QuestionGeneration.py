import logging
import sys
import pandas as pd

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core.evaluation import DatasetGenerator, RelevancyEvaluator
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Response
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import os
import scapy

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

reader = SimpleDirectoryReader(input_files=["Data/icerik.txt"],)

documents = reader.load_data()

data_generator = DatasetGenerator.from_documents(documents)

eval_questions = data_generator.generate_questions_from_nodes()

print(eval_questions)