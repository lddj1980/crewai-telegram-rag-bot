import logging
from dotenv import load_dotenv

from app.container import Container


logging.basicConfig(level=logging.INFO)


def main() -> None:
    load_dotenv()
    container = Container()
    container.bot.run()


if __name__ == "__main__":
    main()
