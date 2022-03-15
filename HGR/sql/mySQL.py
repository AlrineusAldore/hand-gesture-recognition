import sqlite3

con = sqlite3.connect('haha.db')
cur = con.cursor()

class database:
    def __init__(self):
        # Create angles table
        self.creatTable("HAND_DATA", "ANGLES_INDEX INTEGER, EXTREAM_POINTS_INDEX INTEGER, FINGER_COUNT INTEGER,"
                                     " BETEEN_FINGER_POINTS_INDEX INTEGER")
        self.creatTable("ANGLES", "ANGLES_LIST TEXT")
        self.creatTable("EXTREAM_POINTS",  "POINTS_LIST TEXT")
        self.creatTable("BETEEN_FINGER_POINTS",  "POINTS_LIST TEXT")

    def creatTable(self, tableName, columsName):
        cur.execute("CREATE TABLE IF NOT EXISTS " + tableName + " (" + columsName +")")

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
