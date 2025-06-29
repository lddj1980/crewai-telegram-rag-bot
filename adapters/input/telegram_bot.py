import logging
import os
from typing import Callable

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

logger = logging.getLogger(__name__)


class TelegramBot:
    """Telegram bot adapter."""

    def __init__(self, qa_handler: Callable[[str], str]) -> None:
        self.qa_handler = qa_handler
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        # Build the Telegram application and register handlers
        self.app = ApplicationBuilder().token(token).build()
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        self.app.add_handler(CommandHandler("start", self.start))

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        # Greet the user when they invoke /start
        await update.message.reply_text("Olá! Envie sua pergunta.")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        question = update.message.text
        # Send the user's question through the QA service
        answer = self.qa_handler(question)
        await update.message.reply_text(answer)

    def run(self) -> None:
        logger.info("Starting Telegram bot...")
        # Start polling for incoming messages
        self.app.run_polling()
