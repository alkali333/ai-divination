import streamlit as st
import os
from PIL import Image
import secrets
import random

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
    cards = list(major_arcana.keys())

    # Calculate half the length of the entire deck, rounded to the nearest integer
    half_deck_length = round(len(cards) / 2)

    # Choose a random number between 0 and half_deck_length
    num_to_reverse = secrets.randbelow(half_deck_length + 1)

    # Select that many unique cards to reverse from the entire deck
    indices_to_reverse = secrets.sample(cards, num_to_reverse)

    # Reverse the chosen cards by mapping them to a dictionary
    reversed_status = {
        card: " (reversed)" if card in indices_to_reverse else "" for card in cards
    }

    selected_cards = []
    while len(selected_cards) < num:
        card = secrets.choice(cards)
        # Add the card to the selected_cards list along with its reversed status
        selected_cards.append((card, reversed_status[card]))

    return selected_cards


def display_cards(card_info):
    cols = st.columns(3)
    for idx, (card, reversed_status) in enumerate(card_info):
        image_path = f'./images/{major_arcana[card].lower().replace(" ", "-")}.jpg'
        image = Image.open(image_path)
        if reversed_status:
            image = image.rotate(180)  # Rotate the image to display upside down

        with cols[idx]:
            st.image(image, use_column_width=True)
            st.write(f"{major_arcana[card]}{reversed_status}")


def do_reading(card_info, client_question):
    llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0.7, max_tokens=1024)
    card_names = ", ".join(
        [
            f"{major_arcana[card]}{reversed_status}"
            for card, reversed_status in card_info
        ]
    )

    prompt = PromptTemplate(
        template="""Create a dialogue from a fictional mystical and enthusiastic tarot card reader answering a characters questions by reading the cards.
          Address the reader directly. The cards they have chosen are  {card_string}.
          Explain what the cards mean in relation to their question (the individual cards and the cards as a whole).
         You will include some spiritual quotes from tarot related sources. It has to be convincing and will be used in novel.""",
        input_variables=["card_string"],
    )

    formatted_message = prompt.format_prompt(card_string=card_names)

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
            selected_cards = get_random_cards(3)
            display_cards(selected_cards)

            with st.spinner("Reading your cards my dear... "):
                reading = do_reading(selected_cards, user_question)
                st.write(reading)
        else:
            st.error("Please Enter A Question")


# Run the app
if __name__ == "__main__":
    main()
