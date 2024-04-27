from flask import Flask, request, render_template, jsonify


app = Flask(__name__)

import os
from llama_index.core import ServiceContext, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core import load_index_from_storage
from llama_index.core.node_parser import HierarchicalNodeParser
#from gpt4all import GPT4All
from langchain.llms import GPT4All


def build_auto_merging_index(documents,llm, embed_model="local:BAAI/bge-small-en-v1.5",sentence_window_size=3,
                                save_dir="sentence_index",):
    

        # create the hierarchical node parser w/ default settings
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])

    auto_merging_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )
    if not os.path.exists(save_dir):
        auto_merging_index = VectorStoreIndex.from_documents(
            documents, service_context=auto_merging_context
        )
        auto_merging_index.storage_context.persist(persist_dir=save_dir)
    else:
        auto_merging_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=auto_merging_context,
        )

    return auto_merging_index


def get_auto_merging_reranker_engine(auto_merging_index, similarity_top_k=6, rerank_top_n=2):
    
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )

    auto_merging_engine = auto_merging_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[rerank]
    )
    return auto_merging_engine

from llama_index.llms.openai import OpenAI
from data.dataprovider import key
from llama_index.core import SimpleDirectoryReader
#OpenAI.api_key =  key

documents = SimpleDirectoryReader(
    input_files=[r"data/eBook-How-to-Build-a-Career-in-AI.pdf"]
).load_data()

from llama_index.core import Document

document = Document(text="\n\n".join([doc.text for doc in documents]))

index = build_auto_merging_index(
    [document],
    #llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1,api_key=key),
    #llm = GPT4All("mistral-7b-openorca.gguf2.Q4_0.gguf"),
    llm = GPT4All(model=r'C:\Users\91941\.cache\gpt4all\orca-mini-3b-gguf2-q4_0.gguf'), #Replace this path with your model path
    save_dir="./auto_merge_index",
)

query_engine = get_auto_merging_reranker_engine(index, similarity_top_k=6)

def chat_bot_rag(query):
    window_response = query_engine.query(
        query
    )

    return window_response



# Define your Flask routes
@app.route('/')
def home():
    return render_template('bot_1.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['user_input']    
    bot_message = chat_bot_rag(user_message)    
    return jsonify({'response': str(bot_message)})

if __name__ == '__main__':
    #app.run()
    app.run(debug=True)