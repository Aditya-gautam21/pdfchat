import streamlit as st
from htmlTemplates import css, user_template, bot_template

class UserInput:
    def handle_userinput(self, user_question):
        # Get the response with error handling
        try:
            with st.spinner("Analyzing PDF content..."):
                response = st.session_state.conversation({
                    'question': user_question,
                    'chat_history': st.session_state.chat_history
                })
                answer = response['answer'] if isinstance(response, dict) else response
        except Exception as response_e:
            st.error(f"Error getting response: {response_e}")
            answer = f"Error: {response_e}"

        # Update chat history
        st.session_state.chat_history.append((user_question, answer))

        # Display conversation with table formatting preservation
        for q, a in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                # Check if answer contains table-like content and display accordingly
                if any(marker in a for marker in ["|", "===", "Row", "TABLE"]):
                    st.code(a, language=None)
                else:
                    st.markdown(a)
