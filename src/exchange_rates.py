import requests
from typing import Iterable, Dict

def get_exchange_rates(base: str, symbols: Iterable[str]) -> Dict[str, float]:
    url = f"https://api.frankfurter.app/latest"
    params = {
        "from": base,
        "to": ",".join(symbols),
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()
    results = data["rates"]
    output = {}
    for key, value in results.items():
        temp_key = f"{base}/{key}"
        output[temp_key] = float(value)
    return output


if __name__ == "__main__":
    rates = get_exchange_rates("GBP", ["GBP", "USD", "JPY","EUR"])
    print(rates)