import mysql.connector
from mysql.connector import Error

try:
    connection = mysql.connector.connect(
        host="127.0.0.1",
        user="root",              # your MySQL username
        password="Ronaldo=goat7",  # your MySQL password
        database="exoplanet_db"   # your database name
    )

    if connection.is_connected():
        print("✅ Connected successfully to MySQL!")
        cursor = connection.cursor()
        cursor.execute("SHOW TABLES;")
        print("📂 Tables in database:", [t[0] for t in cursor.fetchall()])

except Error as e:
    print(f"❌ Connection failed: {e}")

finally:
    if 'connection' in locals() and connection.is_connected():
        connection.close()
        print("🔒 Connection closed.")
