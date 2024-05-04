import csv
import mysql.connector

# Establish connection to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="mobile_price",
    password="kirito",
    database="Mobiles_prices"
)
cursor = conn.cursor()

# Path to your CSV file
csv_file = '/home/kirito99/Mobile_price_prediction_app/dataset/mobile_price_training_data.csv'

# Open the CSV file and read the data
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)  # Get header row
    insert_query = f"INSERT INTO Mobiles_prices ({', '.join(header)}) VALUES ({', '.join(['%s' for _ in header])})"
    for row in reader:
        # Execute the INSERT statement
        cursor.execute(insert_query, row)


# Commit the transaction
conn.commit()

# Close the cursor and connection
cursor.close()
conn.close()
