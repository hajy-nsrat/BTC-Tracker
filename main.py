import requests

def main():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"

    params = {
        "vs_currency": "usd",
        "days": 365,
        "interval": "daily"
    }

    response = requests.get(url, params=params)
    data = response.json()

    #print(data)
    #print(data.keys())
    print(data['prices'])
    

if __name__ == '__main__':
    main()
