import mysql.connector
from mysql.connector import Error


def main():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="your_username",
            password="your_password"
        )
        cursor = conn.cursor()

        cursor.execute("CREATE DATABASE IF NOT EXISTS user_db")
        cursor.execute("USE user_db")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username VARCHAR(255) PRIMARY KEY,
                password VARCHAR(255) NOT NULL
            )
        ''')

        conn.commit()
        print("✅ 数据库初始化成功")
    except Error as e:
        print(f"❌ 错误: {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()


if __name__ == "__main__":
    main()