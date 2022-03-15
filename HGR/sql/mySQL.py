import sqlite3

con = sqlite3.connect('hehe.db')
cur = con.cursor()

class database:
    def __init__(self):
        # Create angles table
        cur.execute('''CREATE TABLE IF NOT EXISTS angles 
                       (angle real)''')

    def insertData(self, tableName, dataStr):
        # Insert a row of data

        cur.execute("INSERT INTO " + tableName + " VALUES (" + dataStr + ")")

        # Save (commit) the changes
        con.commit()
        #con.close()

    def getData(self, tableName):
        result = []
        for row in cur.execute('SELECT * FROM ' + tableName):
            result.append(row)
        return result
