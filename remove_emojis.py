
import os

def remove_emojis_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Specific replacements for common emojis
        specific_emojis = [
            "[START]", "[UP]", "[DOWN]", "[DATA]", "[GOOD]", "[ALERT]", "", "", "[LIST]", "[TIP]", "[MONEY]", "[HOT]", "[GLOBAL]", "", "", "", "", "", "", "[AI]",
            "[OK]", "[X]", "[WARN]", "", "", "", "", "", "[BANK]", "", "", "[TARGET]", "", "", "", "", "", "", "", "", "", "", "", "", "[FIX]", "", "", "", "",
            "", "", "", "", "[DATE]", "", "[DOWN]", "[UP]", "[CHART]", "", "", "", "", "", "[RANDOM]", "", "", "", "", "", "", "", "", "", ""
        ]

        new_content = content
        for emoji in specific_emojis:
            new_content = new_content.replace(emoji, "")

        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Removed emojis from {file_path}")
        else:
            print(f"No emojis found in {file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    base_dir = "/Users/aavibharucha/Documents/market_ai"
    target_files = [
        "main.py",
        "ai_chatbot.py",
        "market_movers.py",
        "daily_intelligence.py",
        "dashboard.py",
        "watchlist_dashboard.py",
        "custom_dashboard.py",
        "trader_profile.py",
        "paper_trading_ui.py",
        "financial_model_generator.py"
    ]
    
    for file in target_files:
        path = os.path.join(base_dir, file)
        if os.path.exists(path):
            remove_emojis_from_file(path)

if __name__ == "__main__":
    main()
