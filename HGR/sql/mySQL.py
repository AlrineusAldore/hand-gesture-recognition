import sqlite3


#               DATABASE STRUCTURE
#...............................................................................................................................
#:  angles   :       extream_points       : fingers_count :   between_finger_points    :  fingers_lemfings_width_len_list gth  :
#:...........:............................:...............:............................:.......................................:
#: text list : text list of points tuples : int           : text list of points tuples : text list of tuples(width, length)    :
#:...........:............................:...............:............................:..................:....................:



class database:
    def __init__(self):
        self.con = sqlite3.connect('hrtr.db')
        self.cur = self.con.cursor()
        self.creatTable("HAND_DATA",
                        "frame_id int,"
                        "angles text,"
                        "extream_points text,"
                        "fingers_count INTEGER,"
                        "between_points text,"
                        "fings_width_len_list text")

        self.insertData("1, null, null, null, null, null")

    def update(self, colName , strdata):
        self.cur.execute("UPDATE HAND_DATA SET " + colName + " = " + strdata + " WHERE frame_id = 1;")
        self.con.commit()

    def creatTable(self, tableName, columsName):
        self.cur.execute("CREATE TABLE IF NOT EXISTS " + tableName + " (" + columsName +");")
        self.con.commit()

    def insertData(self, dataStr):
        # Insert a row of data

        self.cur.execute("INSERT INTO HAND_DATA VALUES (" + dataStr + ")")

        # Save (commit) the changes
        self.con.commit()
        #con.close()

    def getData(self, tableName):
        result = []
        for row in self.cur.execute('SELECT * FROM ' + tableName):
            result.append(row)
        return result
