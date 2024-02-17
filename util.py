import csv

def check_numeric(x):
    if not isinstance(x, (int, float, complex)):
        raise ValueError(f"{x} is not numeric")

def save_tracking_par_csv(csv_file,xs,ys,ts):
    # Write the coordinates to the CSV file
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['X,pixel', 'Y,pixel', 'Time,sec'])  # Write header
        writer.writerows(list(zip(xs,ys,ts)))  # Write data rows
    print(f"Coordinates saved to {csv_file}")

def read_tracking_par_from_csv(csv_fpath):
    """
    Reads X,Y,time coordinates from csv file
    :param csv_fpath: path to csv path
    :return:
    [X],[Y],[time]
    """
    # Open the CSV file
    with open(csv_fpath, 'r') as f:
        reader = csv.DictReader(f)
        x = []
        y = []
        time = []
        for row in reader:
            x.append(float(row['X,pixel']))
            y.append(float(row['Y,pixel']))
            time.append(float(row['Time,sec']))
    return x,y,time
