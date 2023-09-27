import streamlit as st
from query_against_openSearch import answer_query

# Header/Title of streamlit app
st.markdown("<h1 style='text-align: center; color: red;'>RAG with OpenSearch and Custom Embeddings</h1>",
            unsafe_allow_html=True)


def main():
    """
    The main function orchestrates the various UI elements of the streamlit app.
    """
    if "query_text" not in st.session_state:
        st.session_state["query_text"] = []

    if "query" not in st.session_state:
        st.session_state["query"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []

    with st.container():
        # this is the input text box, where users can submit a question
        input_text = st.text_input("Ask a question:",
                                   "",
                                   key="query_text",
                                   placeholder="What can I help you with?",
                                   on_change=clear_text()
                                   )
        user_input = st.session_state["query"]

        if user_input:
            st.session_state.past.append(user_input)
            # if the user submits a question, it is fed into the answer_query function to return an answer, and writes
            # it to the frontend
            st.write(answer_query(user_input))

    with st.container():
        st.button("clear chat", on_click=clear_session)


def clear_text():
    """
    This is used to clear any text on the screen when clicking the clear text button.
    """
    st.session_state["query"] = st.session_state["query_text"]
    st.session_state["query_text"] = ""


def clear_session():
    """
    This is used to clear any previous inputs stored in the current session.
    """
    for key in st.session_state.keys():
        del st.session_state[key]


if __name__ == "__main__":
    main()
