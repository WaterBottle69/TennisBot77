"""
verify_connection.py — Quick diagnostic to check if the bot is properly connected.

Run with:
    python verify_connection.py

This will:
  1. Check if kalshi_keys.json exists and has valid credentials
  2. Test the Kalshi API connection (fetch balance)
  3. Scan for open tennis markets
  4. Report the authentication status clearly
"""

import asyncio
import json
import os
import sys

async def main():
    print("=" * 60)
    print("  TennisBot77 — Connection Diagnostics")
    print("=" * 60)

    # ── 1. Check credentials file ──────────────────────────────────
    print("\n[1] Checking kalshi_keys.json...")
    keys_path = os.path.join(os.path.dirname(__file__), "kalshi_keys.json")
    if not os.path.exists(keys_path):
        print("  ✗ MISSING — kalshi_keys.json not found!")
        print("    Create it with your API key from: https://kalshi.com/account/security")
        print("    Template:\n")
        print('    {\n      "api_key_id": "YOUR-API-KEY-ID",')
        print('      "private_key_pem": "-----BEGIN RSA PRIVATE KEY-----\\n...\\n-----END RSA PRIVATE KEY-----",')
        print('      "max_bet_usdc": 250.0,')
        print('      "use_prod": false\n    }')
        sys.exit(1)

    with open(keys_path) as f:
        keys = json.load(f)

    api_key = keys.get("api_key_id", "")
    pem     = keys.get("private_key_pem", "")
    use_prod = keys.get("use_prod", False)
    max_bet  = keys.get("max_bet_usdc", 250.0)

    if not api_key or "YOUR_KALSHI" in api_key:
        print("  ✗ api_key_id is still a placeholder — fill in your real key!")
        sys.exit(1)
    if not pem or "YOUR_BASE64" in pem:
        print("  ✗ private_key_pem is still a placeholder — fill in your real private key!")
        sys.exit(1)

    print(f"  ✓ api_key_id : {api_key[:8]}... (redacted)")
    print(f"  ✓ use_prod   : {use_prod}  ({'PRODUCTION' if use_prod else 'DEMO/PAPER TRADING'})")
    print(f"  ✓ max_bet    : ${max_bet:.2f}")

    # ── 2. Load key into cryptography ─────────────────────────────
    print("\n[2] Loading RSA private key...")
    try:
        from config import Config, normalize_kalshi_pem
        from kalshi_client import KalshiClient
        from cryptography.hazmat.primitives import serialization

        clean_pem = normalize_kalshi_pem(pem)
        key = serialization.load_pem_private_key(clean_pem.encode(), password=None)
        key_type = type(key).__name__
        print(f"  ✓ Private key loaded successfully ({key_type})")
    except Exception as e:
        print(f"  ✗ Failed to load private key: {e}")
        print("    Make sure your PEM block is complete and properly formatted.")
        sys.exit(1)

    # ── 3. Test API connection ─────────────────────────────────────
    print("\n[3] Testing Kalshi API connection...")
    config = Config()
    kalshi = KalshiClient(config)

    env_label = "PRODUCTION" if config.KALSHI_USE_PROD else "DEMO"
    print(f"  → Connecting to {env_label}: {config.KALSHI_API_URL}")

    try:
        balance = await kalshi.get_balance()
        if balance > 0 or config.KALSHI_USE_PROD:
            print(f"  ✓ Balance fetched: ${balance:.2f}")
        else:
            print(f"  ✓ Connected (balance=${balance:.2f} — demo accounts start at $0 until funded)")
    except Exception as e:
        print(f"  ✗ API call failed: {e}")
        print("    Check your API key ID and that the key matches your account environment.")
        await kalshi.close()
        sys.exit(1)

    # ── 4. Scan for tennis markets ─────────────────────────────────
    print("\n[4] Scanning for open tennis markets...")
    try:
        matches = await kalshi.get_atp_markets()
        if matches:
            print(f"  ✓ Found {len(matches)} tennis market(s):")
            for m in matches[:5]:
                print(f"    [{m['ticker']}] {m['player_a']} vs {m['player_b']}")
            if len(matches) > 5:
                print(f"    ... and {len(matches)-5} more")
        else:
            print("  ⚠ No open tennis markets found right now.")
            print("    This is normal if no ATP/WTA matches are currently listed on Kalshi.")
    except Exception as e:
        print(f"  ✗ Market scan failed: {e}")

    await kalshi.close()

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STATUS: Bot is properly authenticated ✓")
    if not use_prod:
        print("  MODE:   DEMO (paper trading — no real money)")
        print("  TIP:    Set \"use_prod\": true in kalshi_keys.json to go live.")
    else:
        print("  MODE:   PRODUCTION (real money trading!)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
