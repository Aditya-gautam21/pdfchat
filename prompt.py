from langchain.prompts import PromptTemplate

class Prompt:
    @staticmethod
    def provide_prompt():
        # Question generator prompt
        question_generator_prompt = PromptTemplate(
            input_variables=["question", "chat_history"],
            template="""
Given the chat history and a follow-up question, rephrase the follow-up into a standalone question.
The question should be clear and contain enough context to be understood independently.

Chat history:
{chat_history}

Follow-up question:
{question}

Standalone question:
"""
        )

        # CRITICAL: Much more restrictive prompt
        question_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a PDF document assistant. You must ONLY answer questions using information that is explicitly stated in the provided context below.

STRICT RULES - YOU MUST FOLLOW THESE:
1. If the context does not contain information to answer the question, respond EXACTLY: "I cannot find information about this in the provided PDF documents."
2. NEVER use external knowledge or make assumptions
3. NEVER generate information that is not explicitly stated in the context
4. Quote specific text from the context when possible
5. If you're uncertain, say "I cannot find clear information about this in the PDF"
6. Only reference what is directly written in the context below

Context from PDF documents:
{context}

Question: {question}

Answer (using ONLY the context above - no external knowledge):
"""
        )

        # More restrictive refine prompt
        refine_prompt = PromptTemplate(
            input_variables=["context", "question", "existing_answer"],
            template="""
You are improving an answer using ONLY the additional context provided below.

STRICT RULES:
1. Only use information explicitly stated in the additional context
2. If the additional context doesn't help answer the question, keep the existing answer
3. Never add external knowledge or assumptions
4. Only reference what is directly written in the contexts

Additional Context:
{context}

Question: {question}

Current Answer:
{existing_answer}

Improved Answer (using ONLY the provided contexts):
"""
        )

        return question_generator_prompt, question_prompt, refine_prompt
