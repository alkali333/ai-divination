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

# Mapping from binary tuple to hexagram number in King Wen sequence
hexagram_mapping = {
    (0, 0, 0, 0, 0, 0): 2,
    (1, 0, 0, 0, 0, 0): 24,
    (0, 1, 0, 0, 0, 0): 7,
    (1, 1, 0, 0, 0, 0): 19,
    (0, 0, 1, 0, 0, 0): 15,
    (1, 0, 1, 0, 0, 0): 36,
    (0, 1, 1, 0, 0, 0): 46,
    (1, 1, 1, 0, 0, 0): 11,
    (0, 0, 0, 1, 0, 0): 16,
    (1, 0, 0, 1, 0, 0): 51,
    (0, 1, 0, 1, 0, 0): 40,
    (1, 1, 0, 1, 0, 0): 54,
    (0, 0, 1, 1, 0, 0): 62,
    (1, 0, 1, 1, 0, 0): 55,
    (0, 1, 1, 1, 0, 0): 32,
    (1, 1, 1, 1, 0, 0): 34,
    (0, 0, 0, 0, 1, 0): 8,
    (1, 0, 0, 0, 1, 0): 3,
    (0, 1, 0, 0, 1, 0): 29,
    (1, 1, 0, 0, 1, 0): 60,
    (0, 0, 1, 0, 1, 0): 39,
    (1, 0, 1, 0, 1, 0): 63,
    (0, 1, 1, 0, 1, 0): 48,
    (1, 1, 1, 0, 1, 0): 5,
    (0, 0, 0, 1, 1, 0): 45,
    (1, 0, 0, 1, 1, 0): 17,
    (0, 1, 0, 1, 1, 0): 47,
    (1, 1, 0, 1, 1, 0): 58,
    (0, 0, 1, 1, 1, 0): 31,
    (1, 0, 1, 1, 1, 0): 49,
    (0, 1, 1, 1, 1, 0): 28,
    (1, 1, 1, 1, 1, 0): 43,
    (0, 0, 0, 0, 0, 1): 23,
    (1, 0, 0, 0, 0, 1): 27,
    (0, 1, 0, 0, 0, 1): 4,
    (1, 1, 0, 0, 0, 1): 41,
    (0, 0, 1, 0, 0, 1): 52,
    (1, 0, 1, 0, 0, 1): 22,
    (0, 1, 1, 0, 0, 1): 18,
    (1, 1, 1, 0, 0, 1): 26,
    (0, 0, 0, 1, 0, 1): 35,
    (1, 0, 0, 1, 0, 1): 21,
    (0, 1, 0, 1, 0, 1): 64,
    (1, 1, 0, 1, 0, 1): 38,
    (0, 0, 1, 1, 0, 1): 56,
    (1, 0, 1, 1, 0, 1): 30,
    (0, 1, 1, 1, 0, 1): 50,
    (1, 1, 1, 1, 0, 1): 14,
    (0, 0, 0, 0, 1, 1): 20,
    (1, 0, 0, 0, 1, 1): 42,
    (0, 1, 0, 0, 1, 1): 59,
    (1, 1, 0, 0, 1, 1): 61,
    (0, 0, 1, 0, 1, 1): 53,
    (1, 0, 1, 0, 1, 1): 37,
    (0, 1, 1, 0, 1, 1): 57,
    (1, 1, 1, 0, 1, 1): 9,
    (0, 0, 0, 1, 1, 1): 12,
    (1, 0, 0, 1, 1, 1): 25,
    (0, 1, 0, 1, 1, 1): 6,
    (1, 1, 0, 1, 1, 1): 10,
    (0, 0, 1, 1, 1, 1): 33,
    (1, 0, 1, 1, 1, 1): 13,
    (0, 1, 1, 1, 1, 1): 44,
    (1, 1, 1, 1, 1, 1): 1,
}


def coin_toss():
    return random.choice([2, 3])  # Tails=2, Heads=3


def get_line():
    total = sum(coin_toss() for _ in range(3))
    if total == 6:  # Old Yin
        return (0, True)  # 0 for yin, True for changing
    elif total == 7:  # Young Yang
        return (1, False)
    elif total == 8:  # Young Yin
        return (0, False)
    elif total == 9:  # Old Yang
        return (1, True)


def hexagram_to_number(hexagram):
    return hexagram_mapping[hexagram]


def hexagram():
    lines = [get_line() for _ in range(6)]
    primary_hexagram = tuple(line[0] for line in lines)
    changing_lines = [i for i, line in enumerate(lines) if line[1]]
    transformed_hexagram = tuple(
        (line[0] if i not in changing_lines else 1 - line[0])
        for i, line in enumerate(lines)
    )
    return primary_hexagram, transformed_hexagram, changing_lines


def do_reading(reading, client_question):
    llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0.7, max_tokens=1024)
    prompt = PromptTemplate(
        template="""You are a mystical I Ching reader answering a client's questions. The client has selected the cards {reading}.
          You will explain what the reading in relation to their question (the hexagram, each changing line, and the transformed hexagram if there is one).
            You will include some spiritual quotes from I Ching related sources.""",
        input_variables=["reading"],
    )
    # prompt
    formatted_message = prompt.format_prompt(reading=reading)

    messages = [
        SystemMessage(content=str(formatted_message)),
        HumanMessage(content=client_question),
    ]

    response = llm(messages)
    return response.content


def main():
    st.title("I Ching Reading App")

    user_question = st.text_input("Enter your question:")

    if st.button("Get my reading"):
        if user_question:
            # Perform the I Ching reading
            primary_hexagram, transformed_hexagram, changing_lines = hexagram()

            # Get the numbers for the hexagrams
            primary_number = hexagram_to_number(primary_hexagram)
            transformed_number = hexagram_to_number(transformed_hexagram)

            # Print the final output
            changing_lines_str = ", ".join(str(line + 1) for line in changing_lines)

            # Print the final output
            if changing_lines:
                changing_lines_str = ", ".join(str(line + 1) for line in changing_lines)
                reading = f"User has drawn Hexagram {primary_number} with changing lines on {changing_lines_str}."

            else:
                reading = (
                    f"User has drawn hexagram {primary_number} with no changing lines."
                )

            # If there's a transformed hexagram, print it
            if changing_lines:
                reading += f"\n The transformed hexagram is {transformed_number}"

            with st.spinner("Interpreting your reading... "):
                reading = do_reading(reading, user_question)
                st.write(reading)


# Run the app
if __name__ == "__main__":
    main()
