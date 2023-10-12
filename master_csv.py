import csv
import os
import pandas as pd
import glob

folder = '/home/nottom/Documents/inference/prediction_csvs/2015'

# with open('master_csv.csv', 'w') as out_file:
#     for file in os.listdir(folder):
#         join_path = os.path.join(folder, file)
#         reader = open(join_path, 'r')
#         content = reader.read()
#         print(content[2:30])

# lala = "/home/nottom/Documents/inference/prediction_csvs/2015/1293_20150620_173320_H6170.wav_predictions.csv"
# print(lala[77:80])

os.chdir('/home/nottom/Documents/inference/prediction_csvs')
with open('2015_master_csv.csv', 'w', newline='') as out_file:
    csv_out = csv.writer(out_file)
    column_names = ['year', 'elev', 'transect', 'dist', 'notata']
    writer = csv.DictWriter(out_file, fieldnames = column_names)
    writer.writeheader()

#H1
    notata = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csvs/2015/**/*H1005.wav_predictions.csv', recursive=True):
        print(file)
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['thresholded']:
            if row == 1:
                notata += 1
        year = file[59:63]
        transect = file[75:77]
        dist = file [77:80]
        elev = '2200'
    listrow = [year, elev, transect, dist, str(notata)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4]])

    notata = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csvs/2015/**/*H1070.wav_predictions.csv', recursive=True):
        print(file)
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['thresholded']:
            if row == 1:
                notata += 1
        year = file[59:63]
        transect = file[75:77]
        dist = file [77:80]
        elev = '2200'
    listrow = [year, elev, transect, dist, str(notata)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4]])

    notata = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csvs/2015/**/*H1170.wav_predictions.csv', recursive=True):
        print(file)
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['thresholded']:
            if row == 1:
                notata += 1
        year = file[59:63]
        transect = file[75:77]
        dist = file [77:80]
        elev = '2200'
    listrow = [year, elev, transect, dist, str(notata)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4]])

#H2
    notata = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csvs/2015/**/*H2005.wav_predictions.csv',
                          recursive=True):
        print(file)
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['thresholded']:
            if row == 1:
                notata += 1
        year = file[59:63]
        transect = file[75:77]
        dist = file [77:80]
        elev = '2200'
    listrow = [year, elev, transect, dist, str(notata)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4]])

    notata = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csvs/2015/**/*H2070.wav_predictions.csv',
                          recursive=True):
        print(file)
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['thresholded']:
            if row == 1:
                notata += 1
        year = file[59:63]
        transect = file[75:77]
        dist = file [77:80]
        elev = '2200'
    listrow = [year, elev, transect, dist, str(notata)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4]])

    notata = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csvs/2015/**/*H2170.wav_predictions.csv',
                          recursive=True):
        print(file)
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['thresholded']:
            if row == 1:
                notata += 1
        year = file[59:63]
        transect = file[75:77]
        dist = file [77:80]
        elev = '2200'
    listrow = [year, elev, transect, dist, str(notata)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4]])

#H3
    notata = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csvs/2015/**/*H3005.wav_predictions.csv',
                          recursive=True):
        print(file)
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['thresholded']:
            if row == 1:
                notata += 1
        year = file[59:63]
        transect = file[75:77]
        dist = file [77:80]
        elev = '2200'
    listrow = [year, elev, transect, dist, str(notata)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4]])

    notata = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csvs/2015/**/*H3070.wav_predictions.csv',
                          recursive=True):
        print(file)
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['thresholded']:
            if row == 1:
                notata += 1
        year = file[59:63]
        transect = file[75:77]
        dist = file [77:80]
        elev = '2200'
    listrow = [year, elev, transect, dist, str(notata)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4]])

    notata = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csvs/2015/**/*H3170.wav_predictions.csv',
                          recursive=True):
        print(file)
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['thresholded']:
            if row == 1:
                notata += 1
        year = file[59:63]
        transect = file[75:77]
        dist = file [77:80]
        elev = '2200'
    listrow = [year, elev, transect, dist, str(notata)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4]])

#H4
    notata = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csvs/2015/**/*H4005.wav_predictions.csv',
                          recursive=True):
        print(file)
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['thresholded']:
            if row == 1:
                notata += 1
        year = file[59:63]
        transect = file[75:77]
        dist = file [77:80]
        elev = '2700'
    listrow = [year, elev, transect, dist, str(notata)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4]])

    notata = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csvs/2015/**/*H4070.wav_predictions.csv',
                          recursive=True):
        print(file)
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['thresholded']:
            if row == 1:
                notata += 1
        year = file[59:63]
        transect = file[75:77]
        dist = file [77:80]
        elev = '2700'
    listrow = [year, elev, transect, dist, str(notata)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4]])

    notata = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csvs/2015/**/*H4170.wav_predictions.csv',
                          recursive=True):
        print(file)
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['thresholded']:
            if row == 1:
                notata += 1
        year = file[59:63]
        transect = file[75:77]
        dist = file [77:80]
        elev = '2700'
    listrow = [year, elev, transect, dist, str(notata)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4]])

#H5
    notata = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csvs/2015/**/*H5005.wav_predictions.csv',
                          recursive=True):
        print(file)
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['thresholded']:
            if row == 1:
                notata += 1
        year = file[59:63]
        transect = file[75:77]
        dist = file [77:80]
        elev = '2700'
    listrow = [year, elev, transect, dist, str(notata)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4]])

    notata = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csvs/2015/**/*H5070.wav_predictions.csv',
                          recursive=True):
        print(file)
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['thresholded']:
            if row == 1:
                notata += 1
        year = file[59:63]
        transect = file[75:77]
        dist = file [77:80]
        elev = '2700'
    listrow = [year, elev, transect, dist, str(notata)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4]])

    notata = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csvs/2015/**/*H5170.wav_predictions.csv',
                          recursive=True):
        print(file)
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['thresholded']:
            if row == 1:
                notata += 1
        year = file[59:63]
        transect = file[75:77]
        dist = file [77:80]
        elev = '2700'
    listrow = [year, elev, transect, dist, str(notata)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4]])

#H6
    notata = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csvs/2015/**/*H6005.wav_predictions.csv',
                          recursive=True):
        print(file)
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['thresholded']:
            if row == 1:
                notata += 1
        year = file[59:63]
        transect = file[75:77]
        dist = file [77:80]
        elev = '2700'
    listrow = [year, elev, transect, dist, str(notata)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4]])

    notata = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csvs/2015/**/*H6070.wav_predictions.csv',
                          recursive=True):
        print(file)
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['thresholded']:
            if row == 1:
                notata += 1
        year = file[59:63]
        transect = file[75:77]
        dist = file [77:80]
        elev = '2700'
    listrow = [year, elev, transect, dist, str(notata)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4]])

    notata = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csvs/2015/**/*H6170.wav_predictions.csv',
                          recursive=True):
        print(file)
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['thresholded']:
            if row == 1:
                notata += 1
        year = file[59:63]
        transect = file[75:77]
        dist = file [77:80]
        elev = '2700'
    listrow = [year, elev, transect, dist, str(notata)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4]])
