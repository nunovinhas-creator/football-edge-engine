import logging
import os

import requests

from src.api.http_retry import post_with_retry

logger = logging.getLogger(__name__)

# Prefixo obrigatório em todas as mensagens, para identificar de imediato
# que o alerta vem do Football Edge Engine (e não de outro bot Telegram).
BRAND_PREFIX = "⚽ FOOTBALL EDGE ENGINE"


def send_telegram_alert(message: str) -> bool:
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token:
        logger.error("Telegram: TELEGRAM_BOT_TOKEN não definido — alerta NÃO enviado.")
        print("⚠️ Notificação omitida: TELEGRAM_BOT_TOKEN não definido.")
        return False

    if not chat_id:
        logger.error("Telegram: TELEGRAM_CHAT_ID não definido — alerta NÃO enviado.")
        print("⚠️ Notificação omitida: TELEGRAM_CHAT_ID não definido.")
        return False

    formatted_message = f"{BRAND_PREFIX}\n\n{message}"

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": formatted_message,
        "parse_mode": "Markdown"
    }

    try:
        response = post_with_retry(url, json=payload, timeout=15)
    except (requests.Timeout, requests.ConnectionError) as e:
        logger.error("Telegram: falha de rede ao enviar alerta: %s", e)
        print(f"❌ Erro de rede ao enviar mensagem para o Telegram: {e}")
        return False
    except Exception as e:
        logger.exception("Telegram: erro inesperado ao enviar alerta")
        print(f"❌ Erro ao enviar mensagem para o Telegram: {e}")
        return False

    if response.status_code == 200:
        logger.info("Telegram: alerta enviado com sucesso (HTTP 200).")
        print("📲 Alerta enviado para o Telegram com sucesso!")
        return True

    # Nunca falhar em silêncio: qualquer erro devolvido pela API do Telegram
    # (chat_id inválido, bot bloqueado, Markdown malformado, etc.) fica
    # sempre visível nos logs e na consola.
    logger.error(
        "Telegram: API respondeu com erro HTTP %d: %s",
        response.status_code, response.text
    )
    print(f"❌ Erro da API do Telegram: HTTP {response.status_code} — {response.text}")
    return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_msg = "🧪 *Mensagem de Teste Local*\nO sistema de alertas está ativo e operacional!"
    ok = send_telegram_alert(test_msg)
    print("Resultado do envio de teste:", "OK" if ok else "FALHOU")
