import csv

# Path to your CSV file
csv_file = '/home/kirito99/Mobile_price_prediction_app/dataset/mobile_price_training_data.csv'

# Read the first line of the CSV file to infer column names and data types
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    columns = next(reader)

# Generate the CREATE TABLE statement dynamically
create_table_sql = f"CREATE TABLE Mobiles_prices (id INT AUTO_INCREMENT PRIMARY KEY, {', '.join([f'{column} VARCHAR(255)' for column in columns])});"

print(create_table_sql)
