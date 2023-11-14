import csv
import os
import pandas as pd
import glob

folder = '/home/nottom/Documents/inference/prediction_csv/2021'

os.chdir('/home/nottom/Documents/inference/prediction_csv/2021')
with open('2021_master_csv.csv', 'w', newline='') as out_file:
    csv_out = csv.writer(out_file)
    column_names = ['year', 'elev', 'transect', 'dist', 'ON', 'CW', 'CB', 'CA', 'CT', 'CL']
    writer = csv.DictWriter(out_file, fieldnames = column_names)
    writer.writeheader()

#H1
    ON = 0
    CW = 0
    CB = 0
    CA = 0
    CT = 0
    CL = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csv/2021/**/*H1005.wav_predictions.csv', recursive=True):
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['prediction_0']:
            if row == 1:
                ON += 1
        for row in df['prediction_1']:
            if row == 1:
                CW += 1
        for row in df['prediction_2']:
            if row == 1:
                CB += 1
        for row in df['prediction_3']:
            if row == 1:
                CA += 1
        for row in df['prediction_4']:
            if row == 1:
                CT += 1
        for row in df['prediction_5']:
            if row == 1:
                CL += 1
        year = file[58:62]
        transect = file[74:76]
        dist = file [76:79]
        elev = '2200'
    listrow = [year, elev, transect, dist, str(ON), str(CW), str(CB), str(CA), str(CT), str(CL)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4], listrow[5], listrow[6], listrow[7]
                      , listrow[8], listrow[9]])

    ON = 0
    CW = 0
    CB = 0
    CA = 0
    CT = 0
    CL = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csv/2021/**/*H1070.wav_predictions.csv', recursive=True):
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['prediction_0']:
            if row == 1:
                ON += 1
        for row in df['prediction_1']:
            if row == 1:
                CW += 1
        for row in df['prediction_2']:
            if row == 1:
                CB += 1
        for row in df['prediction_3']:
            if row == 1:
                CA += 1
        for row in df['prediction_4']:
            if row == 1:
                CT += 1
        for row in df['prediction_5']:
            if row == 1:
                CL += 1
        year = file[58:62]
        transect = file[74:76]
        dist = file[76:79]
        elev = '2200'
    listrow = [year, elev, transect, dist, str(ON), str(CW), str(CB), str(CA), str(CT), str(CL)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4], listrow[5], listrow[6], listrow[7]
                         , listrow[8], listrow[9]])

    ON = 0
    CW = 0
    CB = 0
    CA = 0
    CT = 0
    CL = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csv/2021/**/*H1170.wav_predictions.csv', recursive=True):
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['prediction_0']:
            if row == 1:
                ON += 1
        for row in df['prediction_1']:
            if row == 1:
                CW += 1
        for row in df['prediction_2']:
            if row == 1:
                CB += 1
        for row in df['prediction_3']:
            if row == 1:
                CA += 1
        for row in df['prediction_4']:
            if row == 1:
                CT += 1
        for row in df['prediction_5']:
            if row == 1:
                CL += 1
        year = file[58:62]
        transect = file[74:76]
        dist = file[76:79]
        elev = '2200'
    listrow = [year, elev, transect, dist, str(ON), str(CW), str(CB), str(CA), str(CT), str(CL)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4], listrow[5], listrow[6], listrow[7]
                         , listrow[8], listrow[9]])

    #H2
    ON = 0
    CW = 0
    CB = 0
    CA = 0
    CT = 0
    CL = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csv/2021/**/*H2005.wav_predictions.csv',
                          recursive=True):
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['prediction_0']:
            if row == 1:
                ON += 1
        for row in df['prediction_1']:
            if row == 1:
                CW += 1
        for row in df['prediction_2']:
            if row == 1:
                CB += 1
        for row in df['prediction_3']:
            if row == 1:
                CA += 1
        for row in df['prediction_4']:
            if row == 1:
                CT += 1
        for row in df['prediction_5']:
            if row == 1:
                CL += 1
        year = file[58:62]
        transect = file[74:76]
        dist = file[76:79]
        elev = '2200'
    listrow = [year, elev, transect, dist, str(ON), str(CW), str(CB), str(CA), str(CT), str(CL)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4], listrow[5], listrow[6], listrow[7]
                         , listrow[8], listrow[9]])

    ON = 0
    CW = 0
    CB = 0
    CA = 0
    CT = 0
    CL = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csv/2021/**/*H2070.wav_predictions.csv',
                          recursive=True):
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['prediction_0']:
            if row == 1:
                ON += 1
        for row in df['prediction_1']:
            if row == 1:
                CW += 1
        for row in df['prediction_2']:
            if row == 1:
                CB += 1
        for row in df['prediction_3']:
            if row == 1:
                CA += 1
        for row in df['prediction_4']:
            if row == 1:
                CT += 1
        for row in df['prediction_5']:
            if row == 1:
                CL += 1
        year = file[58:62]
        transect = file[74:76]
        dist = file[76:79]
        elev = '2200'
    listrow = [year, elev, transect, dist, str(ON), str(CW), str(CB), str(CA), str(CT), str(CL)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4], listrow[5], listrow[6], listrow[7]
                         , listrow[8], listrow[9]])

    ON = 0
    CW = 0
    CB = 0
    CA = 0
    CT = 0
    CL = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csv/2021/**/*H2170.wav_predictions.csv',
                          recursive=True):
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['prediction_0']:
            if row == 1:
                ON += 1
        for row in df['prediction_1']:
            if row == 1:
                CW += 1
        for row in df['prediction_2']:
            if row == 1:
                CB += 1
        for row in df['prediction_3']:
            if row == 1:
                CA += 1
        for row in df['prediction_4']:
            if row == 1:
                CT += 1
        for row in df['prediction_5']:
            if row == 1:
                CL += 1
        year = file[58:62]
        transect = file[74:76]
        dist = file[76:79]
        elev = '2200'
    listrow = [year, elev, transect, dist, str(ON), str(CW), str(CB), str(CA), str(CT), str(CL)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4], listrow[5], listrow[6], listrow[7]
                         , listrow[8], listrow[9]])

    #H3
    ON = 0
    CW = 0
    CB = 0
    CA = 0
    CT = 0
    CL = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csv/2021/**/*H3005.wav_predictions.csv',
                          recursive=True):
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['prediction_0']:
            if row == 1:
                ON += 1
        for row in df['prediction_1']:
            if row == 1:
                CW += 1
        for row in df['prediction_2']:
            if row == 1:
                CB += 1
        for row in df['prediction_3']:
            if row == 1:
                CA += 1
        for row in df['prediction_4']:
            if row == 1:
                CT += 1
        for row in df['prediction_5']:
            if row == 1:
                CL += 1
        year = file[58:62]
        transect = file[74:76]
        dist = file[76:79]
        elev = '2200'
    listrow = [year, elev, transect, dist, str(ON), str(CW), str(CB), str(CA), str(CT), str(CL)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4], listrow[5], listrow[6], listrow[7]
                         , listrow[8], listrow[9]])

    ON = 0
    CW = 0
    CB = 0
    CA = 0
    CT = 0
    CL = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csv/2021/**/*H3070.wav_predictions.csv',
                          recursive=True):
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['prediction_0']:
            if row == 1:
                ON += 1
        for row in df['prediction_1']:
            if row == 1:
                CW += 1
        for row in df['prediction_2']:
            if row == 1:
                CB += 1
        for row in df['prediction_3']:
            if row == 1:
                CA += 1
        for row in df['prediction_4']:
            if row == 1:
                CT += 1
        for row in df['prediction_5']:
            if row == 1:
                CL += 1
        year = file[58:62]
        transect = file[74:76]
        dist = file[76:79]
        elev = '2200'
    listrow = [year, elev, transect, dist, str(ON), str(CW), str(CB), str(CA), str(CT), str(CL)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4], listrow[5], listrow[6], listrow[7]
                         , listrow[8], listrow[9]])

    ON = 0
    CW = 0
    CB = 0
    CA = 0
    CT = 0
    CL = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csv/2021/**/*H3170.wav_predictions.csv',
                          recursive=True):
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['prediction_0']:
            if row == 1:
                ON += 1
        for row in df['prediction_1']:
            if row == 1:
                CW += 1
        for row in df['prediction_2']:
            if row == 1:
                CB += 1
        for row in df['prediction_3']:
            if row == 1:
                CA += 1
        for row in df['prediction_4']:
            if row == 1:
                CT += 1
        for row in df['prediction_5']:
            if row == 1:
                CL += 1
        year = file[58:62]
        transect = file[74:76]
        dist = file[76:79]
        elev = '2200'
    listrow = [year, elev, transect, dist, str(ON), str(CW), str(CB), str(CA), str(CT), str(CL)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4], listrow[5], listrow[6], listrow[7]
                         , listrow[8], listrow[9]])

    #H4
    ON = 0
    CW = 0
    CB = 0
    CA = 0
    CT = 0
    CL = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csv/2021/**/*H4005.wav_predictions.csv',
                          recursive=True):
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['prediction_0']:
            if row == 1:
                ON += 1
        for row in df['prediction_1']:
            if row == 1:
                CW += 1
        for row in df['prediction_2']:
            if row == 1:
                CB += 1
        for row in df['prediction_3']:
            if row == 1:
                CA += 1
        for row in df['prediction_4']:
            if row == 1:
                CT += 1
        for row in df['prediction_5']:
            if row == 1:
                CL += 1
        year = file[58:62]
        transect = file[74:76]
        dist = file[76:79]
        elev = '2700'
    listrow = [year, elev, transect, dist, str(ON), str(CW), str(CB), str(CA), str(CT), str(CL)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4], listrow[5], listrow[6], listrow[7]
                         , listrow[8], listrow[9]])

    ON = 0
    CW = 0
    CB = 0
    CA = 0
    CT = 0
    CL = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csv/2021/**/*H4070.wav_predictions.csv',
                          recursive=True):
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['prediction_0']:
            if row == 1:
                ON += 1
        for row in df['prediction_1']:
            if row == 1:
                CW += 1
        for row in df['prediction_2']:
            if row == 1:
                CB += 1
        for row in df['prediction_3']:
            if row == 1:
                CA += 1
        for row in df['prediction_4']:
            if row == 1:
                CT += 1
        for row in df['prediction_5']:
            if row == 1:
                CL += 1
        year = file[58:62]
        transect = file[74:76]
        dist = file[76:79]
        elev = '2700'
    listrow = [year, elev, transect, dist, str(ON), str(CW), str(CB), str(CA), str(CT), str(CL)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4], listrow[5], listrow[6], listrow[7]
                         , listrow[8], listrow[9]])

    ON = 0
    CW = 0
    CB = 0
    CA = 0
    CT = 0
    CL = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csv/2021/**/*H4170.wav_predictions.csv',
                          recursive=True):
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['prediction_0']:
            if row == 1:
                ON += 1
        for row in df['prediction_1']:
            if row == 1:
                CW += 1
        for row in df['prediction_2']:
            if row == 1:
                CB += 1
        for row in df['prediction_3']:
            if row == 1:
                CA += 1
        for row in df['prediction_4']:
            if row == 1:
                CT += 1
        for row in df['prediction_5']:
            if row == 1:
                CL += 1
        year = file[58:62]
        transect = file[74:76]
        dist = file[76:79]
        elev = '2700'
    listrow = [year, elev, transect, dist, str(ON), str(CW), str(CB), str(CA), str(CT), str(CL)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4], listrow[5], listrow[6], listrow[7]
                         , listrow[8], listrow[9]])

    #H5
    ON = 0
    CW = 0
    CB = 0
    CA = 0
    CT = 0
    CL = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csv/2021/**/*H5005.wav_predictions.csv',
                          recursive=True):
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['prediction_0']:
            if row == 1:
                ON += 1
        for row in df['prediction_1']:
            if row == 1:
                CW += 1
        for row in df['prediction_2']:
            if row == 1:
                CB += 1
        for row in df['prediction_3']:
            if row == 1:
                CA += 1
        for row in df['prediction_4']:
            if row == 1:
                CT += 1
        for row in df['prediction_5']:
            if row == 1:
                CL += 1
        year = file[58:62]
        transect = file[74:76]
        dist = file[76:79]
        elev = '2700'
    listrow = [year, elev, transect, dist, str(ON), str(CW), str(CB), str(CA), str(CT), str(CL)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4], listrow[5], listrow[6], listrow[7]
                         , listrow[8], listrow[9]])

    ON = 0
    CW = 0
    CB = 0
    CA = 0
    CT = 0
    CL = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csv/2021/**/*H5070.wav_predictions.csv',
                          recursive=True):
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['prediction_0']:
            if row == 1:
                ON += 1
        for row in df['prediction_1']:
            if row == 1:
                CW += 1
        for row in df['prediction_2']:
            if row == 1:
                CB += 1
        for row in df['prediction_3']:
            if row == 1:
                CA += 1
        for row in df['prediction_4']:
            if row == 1:
                CT += 1
        for row in df['prediction_5']:
            if row == 1:
                CL += 1
        year = file[58:62]
        transect = file[74:76]
        dist = file[76:79]
        elev = '2700'
    listrow = [year, elev, transect, dist, str(ON), str(CW), str(CB), str(CA), str(CT), str(CL)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4], listrow[5], listrow[6], listrow[7]
                         , listrow[8], listrow[9]])

    ON = 0
    CW = 0
    CB = 0
    CA = 0
    CT = 0
    CL = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csv/2021/**/*H5170.wav_predictions.csv',
                          recursive=True):
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['prediction_0']:
            if row == 1:
                ON += 1
        for row in df['prediction_1']:
            if row == 1:
                CW += 1
        for row in df['prediction_2']:
            if row == 1:
                CB += 1
        for row in df['prediction_3']:
            if row == 1:
                CA += 1
        for row in df['prediction_4']:
            if row == 1:
                CT += 1
        for row in df['prediction_5']:
            if row == 1:
                CL += 1
        year = file[58:62]
        transect = file[74:76]
        dist = file[76:79]
        elev = '2700'
    listrow = [year, elev, transect, dist, str(ON), str(CW), str(CB), str(CA), str(CT), str(CL)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4], listrow[5], listrow[6], listrow[7]
                         , listrow[8], listrow[9]])

    #H6
    ON = 0
    CW = 0
    CB = 0
    CA = 0
    CT = 0
    CL = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csv/2021/**/*H6005.wav_predictions.csv',
                          recursive=True):
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['prediction_0']:
            if row == 1:
                ON += 1
        for row in df['prediction_1']:
            if row == 1:
                CW += 1
        for row in df['prediction_2']:
            if row == 1:
                CB += 1
        for row in df['prediction_3']:
            if row == 1:
                CA += 1
        for row in df['prediction_4']:
            if row == 1:
                CT += 1
        for row in df['prediction_5']:
            if row == 1:
                CL += 1
        year = file[58:62]
        transect = file[74:76]
        dist = file[76:79]
        elev = '2700'
    listrow = [year, elev, transect, dist, str(ON), str(CW), str(CB), str(CA), str(CT), str(CL)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4], listrow[5], listrow[6], listrow[7]
                         , listrow[8], listrow[9]])

    ON = 0
    CW = 0
    CB = 0
    CA = 0
    CT = 0
    CL = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csv/2021/**/*H6070.wav_predictions.csv',
                          recursive=True):
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['prediction_0']:
            if row == 1:
                ON += 1
        for row in df['prediction_1']:
            if row == 1:
                CW += 1
        for row in df['prediction_2']:
            if row == 1:
                CB += 1
        for row in df['prediction_3']:
            if row == 1:
                CA += 1
        for row in df['prediction_4']:
            if row == 1:
                CT += 1
        for row in df['prediction_5']:
            if row == 1:
                CL += 1
        year = file[58:62]
        transect = file[74:76]
        dist = file[76:79]
        elev = '2700'
    listrow = [year, elev, transect, dist, str(ON), str(CW), str(CB), str(CA), str(CT), str(CL)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4], listrow[5], listrow[6], listrow[7]
                         , listrow[8], listrow[9]])

    ON = 0
    CW = 0
    CB = 0
    CA = 0
    CT = 0
    CL = 0
    for file in glob.glob('/home/nottom/Documents/inference/prediction_csv/2021/**/*H6170.wav_predictions.csv',
                          recursive=True):
        join_path = os.path.join(folder, file)
        df = pd.read_csv(join_path)
        for row in df['prediction_0']:
            if row == 1:
                ON += 1
        for row in df['prediction_1']:
            if row == 1:
                CW += 1
        for row in df['prediction_2']:
            if row == 1:
                CB += 1
        for row in df['prediction_3']:
            if row == 1:
                CA += 1
        for row in df['prediction_4']:
            if row == 1:
                CT += 1
        for row in df['prediction_5']:
            if row == 1:
                CL += 1
        year = file[58:62]
        transect = file[74:76]
        dist = file[76:79]
        elev = '2700'
    listrow = [year, elev, transect, dist, str(ON), str(CW), str(CB), str(CA), str(CT), str(CL)]
    csv_out.writerow([listrow[0], listrow[1], listrow[2], listrow[3], listrow[4], listrow[5], listrow[6], listrow[7]
                         , listrow[8], listrow[9]])

