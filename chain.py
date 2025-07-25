import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents.refine import RefineDocumentsChain
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.llm import LLMChain
from ollama_llm import OllamaLLM
from prompt import Prompt

class Chain:
    def get_conversation_chain(self, vectorstore):
        llm = OllamaLLM(model="mistral")

        chat_history = StreamlitChatMessageHistory()

        # Memory buffer for chat history
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            chat_memory = chat_history,
            output_key="answer"
        )

        question_generator_prompt, question_prompt, refine_prompt = Prompt.provide_prompt()
        question_generator = LLMChain(llm=llm, prompt=question_generator_prompt)

        qa_chain = load_qa_chain(
            llm=llm,
            chain_type="refine",
            question_prompt=question_prompt,   
            refine_prompt=refine_prompt,
            document_variable_name="context"
        )

        # Conversational RAG chain
        conversation_chain = ConversationalRetrievalChain(
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                memory=memory,
                question_generator=question_generator,
                combine_docs_chain=qa_chain,
                return_source_documents=True
            )
        return conversation_chain
