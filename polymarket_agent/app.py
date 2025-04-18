import logging
import os

from dotenv import load_dotenv
from flask import Flask

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")


def get_actual_news():
    pass


def get_actual_tweets():
    pass


def get_actual_bets():
    pass


def get_yes_prob():
    pass


def make_bet():
    pass


app = Flask(__name__)


if __name__ == "__main__":
    app.run(debug=True, port=5001)


