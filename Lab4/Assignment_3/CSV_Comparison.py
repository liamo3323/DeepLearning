import csv

def compare_csv_values(csv1, csv2, input):
    csv1_data = {}
    csv2_data = {}

    matchA = ""
    matchB = ""

    with open(csv1, 'r') as file1:
        reader1 = csv.reader(file1)
        csv1_data = [row for row in reader1]

    with open(csv2, 'r') as file2:
        reader2 = csv.reader(file2)
        csv2_data = [row for row in reader2]

    matching_rows = []
    for row in csv1_data:
        if input in row:
            matchA = row

    for row in csv2_data:
        if input in row:
            matchB = row

    if matchA != matchB:
        print("Mismatched rows: ")
        print("ViT: ", matchA)
        print("Main: ", matchB)

if __name__ == "__main__":

    ViT_csv = "ViT_Model/66406106.csv"
    Main_csv = "res.csv"

    for group_num in range(50):
        for image_num in range(10):
            compare_csv_values(ViT_csv, Main_csv, "group"+str(group_num)+"/images/"+str(image_num)+".jpg")