import streamlit as st
from htmlTemplates import css, user_template, bot_template

class UserInput:
    def handle_userinput(self, user_question):
        
        response = st.session_state.conversation({
                'question': user_question,
                'chat_history': st.session_state.chat_history
            })
        
        answer = response['answer'] if isinstance(response, dict) else response

        # Update history
        st.session_state.chat_history.append((user_question, answer))

        for q, a in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                st.markdown(a)
                

    def get_conversation_history(self):
        with st.expander("Conversation History"):
            for i, (q, a) in enumerate(st.session_state.chat_history):
                    st.markdown(f"""
                **Q{i+1}:** {q}  
                **A{i+1}:** {a}
                ---
                """)