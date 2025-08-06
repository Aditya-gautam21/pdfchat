from langchain.prompts import PromptTemplate

class Prompt:
    @staticmethod
    def provide_enhanced_prompts():
        """Provide enhanced prompts optimized for table and image content"""
        
        # Enhanced question generator
        question_generator_prompt = PromptTemplate(
            input_variables=["question", "chat_history"],
            template="""
Given the chat history and a follow-up question, rephrase the follow-up into a clear, standalone question.
The question should preserve any specific requests for tables, data, images, or numerical information.

Chat history:
{chat_history}

Follow-up question:
{question}

Standalone question (preserve requests for tables, data, images, or numbers):
"""
        )

        # Enhanced QA prompt with table/image focus
        qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are an expert PDF document assistant. You must ONLY answer using information explicitly provided in the context below.

CRITICAL RULES:
1. If the context doesn't contain the requested information, respond: "I cannot find this information in the provided PDF documents."
2. NEVER use external knowledge or make assumptions
3. NEVER generate information not explicitly stated in the context
4. Always cite specific content from the context when possible

SPECIAL HANDLING FOR STRUCTURED DATA:
- When you see content marked with "STRUCTURED_DATA_START" and "STRUCTURED_DATA_END", this is table data
- When you see content with "HEADERS:" and "DATA_ROW", this is formatted table content
- When you see content marked with "IMAGE" sections, this is text extracted from images
- Preserve the structure and formatting of tables in your response
- For numerical queries, provide exact values from the structured data
- When referencing tables, mention the page number and table number if available

RESPONSE FORMAT:
- For table queries: Present data in clear, structured format
- For image queries: Describe what text/content was found in images  
- For overview queries: Provide comprehensive summary using all available content
- Always specify the source (page number, table number, image number) when possible

Context from PDF documents:
{context}

Question: {question}

Answer (using ONLY the context above, with special attention to structured data):
"""
        )

        # Enhanced refine prompt
        refine_prompt = PromptTemplate(
            input_variables=["context", "question", "existing_answer"],
            template="""
You are improving an answer using additional context. Use ONLY the information provided.

RULES:
1. Only use information explicitly stated in the additional context
2. If additional context doesn't help, keep existing answer unchanged
3. Pay special attention to any STRUCTURED_DATA, TABLE, or IMAGE content
4. Merge information logically, preserving table structure and image content
5. Never add external knowledge

Additional Context:
{context}

Question: {question}

Current Answer:
{existing_answer}

Improved Answer (incorporating any structured data from additional context):
"""
        )

        return question_generator_prompt, qa_prompt, refine_prompt


# Usage note: Update your import in the original chain.py to use these enhanced versions
# from prompt import EnhancedPrompt as Prompt