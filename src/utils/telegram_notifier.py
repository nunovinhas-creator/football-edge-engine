import os
import json
import urllib.request

def send_telegram_alert(message: str) -> bool:
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        print("⚠️ Notificação omitida: TELEGRAM_BOT_TOKEN ou TELEGRAM_CHAT_ID não definidos.")
        return False

    # Prefix obrigatório no topo para diferenciar este projeto dos outros repos
    formatted_message = f"🚨 *NOVO SISTEMA TESTE*\n────────────────────\n\n{message}"

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": formatted_message,
        "parse_mode": "Markdown"
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                print("📲 Alerta enviado para o Telegram com sucesso!")
                return True
    except Exception as e:
        print(f"❌ Erro ao enviar mensagem para o Telegram: {e}")
        return False

if __name__ == "__main__":
    test_msg = "🧪 *Mensagem de Teste Local*\nO sistema de alertas está ativo e operacional!"
    send_telegram_alert(test_msg)
