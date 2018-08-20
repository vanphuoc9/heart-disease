import csv
row = [63,1,1,145,233,1,2,150,0,2.3,3,0,6,0]

with open('Heart_Disease_Data.csv', 'a') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(row)

csvFile.close()
