import pandas as pd
import yfinance as yf
import random
import csv

anzahl = 0

 
def write_csv(filename, data, header=None):
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        if header:
            writer.writerow(header)
        writer.writerows(data)

def split_csv(filename, num_rows, has_header=True):
    name, extension = filename.split('.')
    file_no = 1
    chunk = []
    row_count = 0
    header = ''
 
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if has_header:
                header = row
                has_header = False
                continue
            chunk.append(row)
            row_count += 1
            if row_count >= num_rows:
                write_csv(f'{name}-{file_no}.{extension}', chunk, header)
                chunk = []
                file_no += 1
                row_count = 0
        if chunk:
            write_csv(f'{name}-{file_no}.{extension}', chunk, header)



for i in range(anzahl):
    data = pd.read_csv("data/listSymbols.csv")
    symbols = data["Symbol"]
    choice = random.choice(symbols)
    print(choice)
    ticker = yf.Ticker(choice)
    max = ticker.history("max")
    print(max.columns.tolist())
    max.info()
    max.to_csv(f"data/{choice}.csv")
    split_csv(f"data/{choice}.csv", 1000)
    
