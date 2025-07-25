from langchain.prompts import PromptTemplate

class Prompt:
    @staticmethod
    def provide_prompt():
        # Question generator prompt (ONLY 'question')
        question_generator_prompt = PromptTemplate(
        input_variables=["question", "chat_history"],
        template="""
        Given the chat history and a follow-up question, rephrase the follow-up into a standalone question.

        Chat history:
        {chat_history}

        Follow-up question:
        {question}

        Standalone question:
        """
        )


        # Initial answering prompt (NEEDS 'context' + 'question')
        question_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
Use the following context to answer the question clearly and directly but do not make it too short.

Context:
{context}

Question:
{question}

Answer:"""
        )

        # Refine prompt (NEEDS 'context' + 'question' + 'existing_answer')
        refine_prompt = PromptTemplate(
            input_variables=["context", "question", "existing_answer"],
            template="""
Improve the existing answer using the additional context below if necessary and only return the answer.

Context:
{context}

Question:
{question}

Existing Answer:
{existing_answer}

Improved Answer:"""
        )

        return question_generator_prompt, question_prompt, refine_prompt
