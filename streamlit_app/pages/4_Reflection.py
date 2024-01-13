import streamlit as st

st.set_page_config(
    page_title="Reflection and Challenges",
    page_icon="🤔",
)

st.write("# Reflection and Challenges 👌")

# st.sidebar.success("Select a demo above.")

st.markdown(
    """
    ### Challenges
    Since we are students, we had a lot of classwork to do and exams coming up while we were doing this project, \
    making scheduling times to work on it and meet together very challenging. Most of us were brand new to these new machine learning \
    tools so we had to learn a lot as we went through it. The datasets that were provided online were very skewed for the I personality types, \
    making the model overfitted and giving bad accuracy initially. 

    ### Reflection
    We believe our current project could be further developed into making a more responsive, personalized chatbot. \
    We had trouble making progress on the project due to scheduling conflicts, and having multiple meetings a week proved difficult. \
    We believe future projects could benefit from smaller group sizes or groups based on similar interests.
    
    Also, we could've tried to RoBERTa model, which was trained on a larger set of text corpus -- this might give us a better training result.
        
"""
)