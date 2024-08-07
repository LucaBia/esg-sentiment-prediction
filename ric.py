import pandas as pd

# Descargar la lista de componentes del S&P 500 usando yfinance
sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Leer la tabla directamente desde Wikipedia
sp500_table = pd.read_html(sp500_url)[0]

# Extraer los símbolos (equivalente a RICs)
sp500_symbols = sp500_table['Symbol'].tolist()

# Mostrar los símbolos de los componentes del S&P 500
print("Símbolos de los componentes del S&P 500:")
for symbol in sp500_symbols:
    print(symbol)