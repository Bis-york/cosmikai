import mysql.connector
from mysql.connector import Error


# -----------------------------
# Create a reusable DB connection
# -----------------------------
def create_connection():
    try:
        connection = mysql.connector.connect(
            host="127.0.0.1",         # your MySQL server (or IP if remote)
            user="root",              # your MySQL username
            password="Ronaldo=goat7",  # your MySQL password
            database="exoplanet_db"   # ✅ updated database name
        )
        return connection
    except Error as e:
        print(f"❌ Error while connecting to MySQL: {e}")
        return None


# -----------------------------
# Fetch existing result by star_id
# -----------------------------
def get_result_by_star_id(star_id):
    conn = create_connection()
    if not conn:
        return None

    cursor = conn.cursor(dictionary=True)
    query = "SELECT * FROM exoplanet_results WHERE star_id = %s"  # ✅ updated table name
    cursor.execute(query, (star_id,))
    result = cursor.fetchone()

    cursor.close()
    conn.close()
    return result


# -----------------------------
# Insert new result into database
# -----------------------------
def insert_result(star_id, result_json):
    conn = create_connection()
    if not conn:
        return False

    cursor = conn.cursor()
    query = "INSERT INTO exoplanet_results (star_id, result_json) VALUES (%s, %s)"  # ✅ updated table name
    cursor.execute(query, (star_id, str(result_json)))
    conn.commit()

    cursor.close()
    conn.close()
    return True
