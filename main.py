import streamlit as st
import os
import random
from PIL import Image


from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage

from dotenv import load_dotenv

# Attempt to load the .env file if it exists
env_path = ".env"
if os.path.isfile(env_path):
    load_dotenv(dotenv_path=env_path)

# Check if OPENAI_KEY is set in environment variables
# If not, set it using Streamlit secrets
if "OPENAI_API_KEY" not in os.environ:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    else:
        raise EnvironmentError(
            "Open AI key not found in environment variables or Streamlit secrets."
        )


# Major Arcana dictionary with plain English names
major_arcana = {
    0: "The Fool",
    1: "The Magician",
    2: "The High Priestess",
    3: "The Empress",
    4: "The Emperor",
    5: "The Hierophant",
    6: "The Lovers",
    7: "The Chariot",
    8: "Strength",
    9: "The Hermit",
    10: "Wheel of Fortune",
    11: "Justice",
    12: "The Hanged Man",
    13: "Death",
    14: "Temperance",
    15: "The Devil",
    16: "The Tower",
    17: "The Star",
    18: "The Moon",
    19: "The Sun",
    20: "Judgement",
    21: "The World",
}


def get_random_cards(num):
    return random.sample(list(major_arcana.keys()), num)


def display_cards(card_info):
    cols = st.columns(3)
    for idx, (image, name) in enumerate(card_info):
        with cols[idx]:
            st.image(image, use_column_width=True)
            st.write(name)


def do_reading(card_string, client_question):
    llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0.7, max_tokens=1024)
    prompt = PromptTemplate(
        template="""You are a mystical tarot reader answering a client's questions. The client has selected the cards {card_string}.
          You will explain what the cards mean in relation to their question (the individual cards and the cards as a whole).
            You will include some spiritual quotes from tarot related sources.""",
        input_variables=["card_string"],
    )
    # prompt
    formatted_message = prompt.format_prompt(card_string=card_string)

    messages = [
        SystemMessage(content=str(formatted_message)),
        HumanMessage(content=client_question),
    ]

    response = llm(messages)
    return response.content


def main():
    st.title("Tarot Reading App")

    user_question = st.text_input("Enter your question:")

    if st.button("Get my reading"):
        if user_question:
            selected_keys = get_random_cards(3)
            card_info = [
                (
                    Image.open(
                        f'./images/{major_arcana[key].lower().replace(" ", "-")}.jpg'
                    ),
                    major_arcana[key],
                )
                for key in selected_keys
            ]
            display_cards(card_info)

            card_names = ", ".join([major_arcana[key] for key in selected_keys])
            # Display the combined string of card names
            st.write(card_names)

            with st.spinner("Reading your cards my dear... "):
                reading = do_reading(card_names, user_question)
                st.write(reading)
        else:
            print("Please Enter A Question")


# Run the app
if __name__ == "__main__":
    main()
