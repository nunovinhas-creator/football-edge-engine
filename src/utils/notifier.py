def chunk_message(message: str, max_chars: int = 4000) -> list:
    """
    Divide mensagens longas em vários chunks para evitar erros na API do Telegram.
    """
    return [message[i:i + max_chars] for i in range(0, len(message), max_chars)]

def format_telegram_alert(match_name: str, ev_data: dict) -> str:
    """
    Formata o alerta num formato limpo (Markdown-friendly) para poupar espaço.
    """
    lines = [f"🚨 **Value Alert: {match_name}** 🚨", ""]
    for market, data in ev_data.items():
        if data['ev'] > 0:
            lines.append(f"✅ {market.upper()}: Odd {data['odd']} | EV {data['ev']:.2%} | Stake {data['stake_pct']:.2%}")
    return "\n".join(lines)
