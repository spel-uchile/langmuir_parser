import argparse
import ephem
from ephem import degree
import pandas as pd
import numpy as np

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.dates as mdates
import datetime
import seaborn as sns

def get_parameters():
    """
    Parse command line parameters
    :return: namespace with parameters
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('files', type=str, nargs='+', help="CSV Files to process")
    parser.add_argument('-c, --concatenate', dest="concatenate", action="store_true", help="Concatenate all files in one")
    parser.add_argument('-a, --animate', dest="animate", action="store_true", help="Show an animated plot")
    parser.add_argument('-s, --save', dest="save", action="store_true", help="Save figure")
    parser.add_argument('-l', '--list', dest="color_min_max", nargs=6, help='Colobar range min and max for each graphic')

    return parser.parse_args()


def get_tle(date=None):
    """
    Return the latest or given date TLE as three elements array
    :param date:
    :return: Array with the satellite name and the two TLE lines
    """
    tle = ["SUCHAI",
           "1 42788U 17036Z   18216.85523299  .00000965  00000-0  45215-4 0  9998",
           "2 42788  97.3976 275.9462 0010213 264.9242  95.0828 15.22050052 61992"]

    return tle


def add_long_lat(dataset, tle):
    """
    Appends the longitued and latitude columns to a dataset that contains
    a "time" column (date and time) using the given TLE.
    :param dataset: DataFrame Dataset with a "time" column
    :param tle: Array Three elements array with TLE data
    :return: DataFrame with Lat and Long columns added
    """
    tle_rec = ephem.readtle(tle[0], tle[1], tle[2])
    dates = dataset["time"]
    lat = []
    long = []
    for date in dates:
        tle_rec.compute(date)
        long.append(tle_rec.sublong/degree)
        lat.append(tle_rec.sublat/degree)

    dataset["Lon"] = long
    dataset["Lat"] = lat

    return dataset

def add_plasma(dateset):
    """

    :param dateset:
    :return:

    Plasma current = 0.004723*(J6^3)*exp(((1.602176565*10^(-19))/(1*(1.3806488*10^(-23))*J6))*(I6-(H6/11.59991)+3.0082)/(12.07465+(1/11.59991))-((1.85847675*10^(-19))/(1.3806488*10^(-23)*J6)))
    Electron densi = (M6/(1.602176565*10^(-19)*4*pi()*(0.0048^2)))*sqrt((2*PI()*9.10938356*10^(-31))/(1.3806488*10^(-23)*300))
    """

    dataset["Plasma current"] = 0.004723*(dataset["Plasma temperature"]**3)*np.exp(
        ((1.602176565 * 10**(-19)) / (1 * (1.3806488 * 10**(-23)) * dataset["Plasma temperature"])) * (
                dataset["Plasma voltage"] - (dataset["Sweep voltage"] / 11.59991) + 3.0082) / (
                    12.07465 + (1 / 11.59991)) - (
                    (1.85847675 * 10**(-19)) / (1.3806488 * 10**(-23) * dataset["Plasma temperature"])))

    dataset["Electron density 300K"] = (dataset["Plasma current"]/(1.602176565*10**(-19)*4*np.pi*(0.0048**2)))*np.sqrt((2*np.pi*9.10938356*10**(-31))/(1.3806488*10**(-23)*300))
    dataset["Electron density 3000K"] = (dataset["Plasma current"]/(1.602176565*10**(-19)*4*np.pi*(0.0048**2)))*np.sqrt((2*np.pi*9.10938356*10**(-31))/(1.3806488*10**(-23)*3000))

    return dataset


def plot_map(dataset, title, columns, save, show, minmax_list, min, max):
    """
    Plots each column of a Pandas DataFrame that contains the columns "Long"
    and "Lat" over a Earth map. Optionally saves the plot to disk or just
    displays the plot on the screen.

    :param dataset: pd.DataFrame
    :param title: Str. Dataset title
    :param save: Bool. Save the figure
    :param show: Bool. Show the plot
    :param minmax_list: List. Contains min and max of each graphic
    :param min: Float. Global min to set the colorbar range
    :param max: Float. Global max to set the colorbar range
    :return: None
    """
    # scales = [1, 1e9, 1e-9, 1e-9]
    # min, max = min[columns], max[columns]

    minmax_ind = 0

    for column in columns:
        data = dataset[column]
        #print(data.min())
        #print(data.max())
        # Set world axes
        fig = plt.figure(figsize=(1024 / 96, 768 / 96), dpi=96)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        #ax.stock_img()
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.5,
                          linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        # Plot certain places as black triangles
        plt.scatter(-76.704, -11.739, c="k", marker="^",s=4)  # Jicamarca, Peru
        plt.scatter(-71.488, 42.623, c="k", marker="^", s=4)  # MIT Haystack, EEUU
        plt.scatter(-66.752, 18.344, c="k", marker="^", s=4)  # Arecibo, Puerto Rico
        plt.scatter(-94.829, 74.697, c="k", marker="^", s=4)  # Resolute Bay, Canada
        plt.scatter(-147.488, 65.125, c="k", marker="^", s=4)  # Poker Flat, Alaska

        plt.title(column + "\n" + "".join(title))

        # Plot as scatter
        plt.scatter(dataset["Lon"], dataset["Lat"], c=dataset[column], s=20, cmap="plasma", alpha=0.5, linewidths=0, edgecolors=None)
        colorbar = plt.colorbar()
        colorbar.set_label(column)

        # Set colorbar scale
        if minmax_list:
            plt.clim(minmax_list[minmax_ind * 2], minmax_list[minmax_ind * 2 + 1])
            minmax_ind+=1


        # Finally plot
        if save:
            plt.savefig("".join(title) + "-" + column + ".png", dpi=96)
        if show:
            plt.show()

        plt.close()

def add_is_anomaly(dataset, threshold):

    """
    Appends a column that sorts out the particles counter
    values (greater or lower than a threshold) to a dataset that contains
    a "Particles counter" column.
    :param dataset: DataFrame Dataset with a "Particles counter" column
    :return: DataFrame with is_anomaly column added
    """

    particles = dataset['Particles counter']
    is_anomaly = []

    for row in particles:
        if row >= threshold:
            is_anomaly.append(1)
        else:
            is_anomaly.append(0)

    dataset["is_anom"] = is_anomaly

    return dataset

def add_day(dataset):

    dataset['time'] = pd.to_datetime(dataset['time'], format='%Y-%m-%d %H:%M:%S')

    day_bound = datetime.time(6, 0, 0)
    night_bound = datetime.time(18, 0, 0)

    time = dataset['time']
    day = []

    for row in time:
        if row.time() >= day_bound and row.time() < night_bound:
            day.append(1)
        else:
            day.append(0)

    dataset["day"] = day
    return dataset

"""def min_date(dataset):
    
    print(dataset)
    dataset['time'] = pd.to_datetime(dataset['time'], format='%Y-%m-%d %H:%M:%S')
    dataset['Date'], dataset['Time'] = dataset['time'].dt.normalize(), dataset['time'].dt.time
    dataset = dataset.sort_values(by=['Time'])
    print(dataset.iloc[0])"""

def add_anom_diff(dataset):
    """
    Appends a column that represents the difference between the actual
    and previous row of is_anom column of the dataset.
    :param dataset: DataFrame Dataset with a "is_anom" column
    :return: DataFrame with anom_diff column added
    """

    # time column as datetime
    dataset['time'] = pd.to_datetime(dataset['time'], format='%Y-%m-%d %H:%M:%S')

    # sort dataset by time
    dataset = dataset.sort_values(by=['time'])

    is_anom = dataset['is_anom']
    anom_diff = is_anom.diff()
    dataset['anom_diff'] = anom_diff
    dataset.loc[0,'anom_diff'] = dataset.iloc[0]['is_anom']
    print(dataset.iloc[0]['anom_diff'])
    return dataset


def add_anom_cluster(dataset):
    """
    Appends a column that represents the different clusters
    of values that are equal to 1 on is_anom column.
    :param dataset: DataFrame Dataset with a "is_anom" column
    :return: DataFrame with group column added
    """

    dataset = add_anom_diff(dataset)

    # time column as datetime
    dataset['time'] = pd.to_datetime(dataset['time'], format='%Y-%m-%d %H:%M:%S')

    # sort dataset by time
    dataset = dataset.sort_values(by=['time'])

    anom_diff = dataset['anom_diff']
    group = dataset['is_anom']
    dataset['group'] = group
    group_col_ind = dataset.columns.get_loc('group')
    is_anom_col_ind = dataset.columns.get_loc('is_anom')
    #print(group_col_ind)
    i = 0
    multiplier = 1
    for row in anom_diff:
        if row == 1:
            dataset.iloc[i:, group_col_ind] = dataset.iloc[i:, is_anom_col_ind] * multiplier
            multiplier += 1
        i += 1

    return dataset

def add_season(dataset):

    dataset['time'] = pd.to_datetime(dataset['time'], format='%Y-%m-%d %H:%M:%S')
    time = dataset['time']
    dataset['SEASON'] = time.dt.dayofyear.map(season)
    return dataset

def season(x):

    fall = range(80, 172)
    winter = range(172, 264)
    spring = range(264, 355)

    if x in spring:
        return 3
    if x in winter:
        return 2
    if x in fall:
        return 1
    else:
        return 4

def make_in_max_out(dataset):

    # time column as datetime
    dataset['time'] = pd.to_datetime(dataset['time'], format='%Y-%m-%d %H:%M:%S')

    # sort dataset by time
    dataset = dataset.sort_values(by=['time'])

    is_anom = dataset['is_anom']
    i = 0

    #create empty list and empty datasets
    arr = []
    aux_dataset = pd.DataFrame(columns=dataset.columns.values)
    final_dataset = pd.DataFrame(columns=dataset.columns.values)

    #iterate through rows to find enter, max and out points(of the anomaly)
    while i < is_anom.count():
        while i < is_anom.count() and is_anom.iloc[i] == 1:
            arr.append(dataset.iloc[[i]])
            i += 1
        if len(arr) != 0:

            #concat datasets inside arr to aux_dataset
            final_arr = pd.concat(arr, sort=False)
            aux_dataset = pd.concat([aux_dataset, final_arr], sort=False) #aux_dataset.append(arr)
            #print(aux_dataset)

            #get row with the max number of particles inside
            aux_dataset['Particles counter'] = aux_dataset['Particles counter'].astype('float64')
            penultimate = aux_dataset['Particles counter'].count() - 1
            aux_dataset = aux_dataset.iloc[1:penultimate]
            if not aux_dataset.empty:
                row = aux_dataset.loc[aux_dataset['Particles counter'].idxmax()]
                row = pd.DataFrame([row])
            #detect duplicates
            if aux_dataset.empty:
               if 0 == len(arr) - 1:
                   final_dataset = pd.concat([final_dataset, arr[0]], ignore_index=True, sort=False) #final_dataset.append([arr[0]])
               else:
                   final_dataset = pd.concat([final_dataset, arr[0], arr[len(arr) - 1]], ignore_index=True, sort=False) #final_dataset.append([arr[0], arr[len(arr) - 1]])
            else:
                final_dataset = pd.concat([final_dataset, arr[0], arr[len(arr) - 1], row], ignore_index=True, sort=False) #final_dataset.append([arr[0], arr[len(arr) - 1], row])
            #reset list and aux_dataset
            arr = []
            aux_dataset = pd.DataFrame(columns=dataset.columns.values)
        else:
            i += 1
    return final_dataset

def plot_lat_in_time(dataset, threshold, title, save, show):
    # time column as datetime
    dataset['time'] = pd.to_datetime(dataset['time'], format='%Y-%m-%d %H:%M:%S')

    # sort dataset by time
    dataset = dataset.sort_values(by=['time'])

    # add is_anomaly column
    dataset = add_is_anomaly(dataset, threshold)

    # filter rows where is_anomaly value is True or 1
    in_threshold = dataset['is_anom'] == 1
    dataset = dataset[in_threshold]
    dataset['time'] = pd.to_datetime(dataset['time'], format='%Y-%m-%d %H:%M:%S')

    minmax_ind = 0

    #make columns
    lat = dataset["Lat"]
    time = dataset["time"]

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))

    # plot
    blue_data = dataset.loc[dataset['day'] == 0]
    lat_blue = blue_data["Lat"]
    time_blue = blue_data["time"]

    red_data = dataset.loc[dataset['day'] == 1]
    lat_red = red_data["Lat"]
    time_red = red_data["time"]

    #if not blue_data.empty:
    plt.plot(time_blue, lat_blue, 'ro', markersize=0.5, color = 'blue')
    #if not red_data.empty:
    plt.plot(time_red, lat_red, 'ro', markersize=0.5, color='red')

    # beautify the x-labels
    plt.gcf().autofmt_xdate()
    #_ = plt.xticks(rotation=90)

    #set y axis range and grid
    plt.ylim((-90, 90))
    ax = plt.gca()
    ax.yaxis.grid(True)

    #set labels and title
    plt.xlabel("Time (Y-m-d H:M:S)")
    plt.ylabel("Latitude")
    plt.title("Time vs. Latitude")

    # Finally plot
    if save:
        plt.savefig("".join(title) + "-Lat" + ".png", dpi=96)
    if show:
        plt.show()

    plt.close()
    print(dataset)

def plot_part_in_threshold(dataset, title, columns, save, show, minmax_list, threshold):

    #time column as datetime
    dataset['time'] = pd.to_datetime(dataset['time'], format='%Y-%m-%d %H:%M:%S')

    #sort dataset by time
    dataset = dataset.sort_values(by=['time'])

    #add is_anomaly column
    dataset = add_is_anomaly(dataset, threshold)

    #filter rows where is_anomaly value is True or 1
    in_threshold = dataset['is_anom'] == 1
    dataset = dataset[in_threshold]

    minmax_ind = 0

    #plot
    for column in columns:
        data = dataset[column]
        # Set world axes
        fig = plt.figure(figsize=(1024 / 96, 768 / 96), dpi=96)
        ax = plt.axes(projection=ccrs.PlateCarree())
        plt.xlim(-180, 180)
        plt.ylim(-90, 90)
        ax.coastlines()
        #ax.stock_img()
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                           linewidth=1, color='gray', alpha=0.5,
                           linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER


        # Plot certain places as black triangles
        plt.scatter(-76.704, -11.739, c="k", marker="^",s=4)  # Jicamarca, Peru
        plt.scatter(-71.488, 42.623, c="k", marker="^", s=4)  # MIT Haystack, EEUU
        plt.scatter(-66.752, 18.344, c="k", marker="^", s=4)  # Arecibo, Puerto Rico
        plt.scatter(-94.829, 74.697, c="k", marker="^", s=4)  # Resolute Bay, Canada
        plt.scatter(-147.488, 65.125, c="k", marker="^", s=4)  # Poker Flat, Alaska

        plt.title(column + "\n" + "".join(title))

        if not dataset.empty:
            # Plot as scatter
            plt.scatter(dataset["Lon"], dataset["Lat"], c=dataset[column], s=20, cmap="plasma", alpha=0.5, linewidths=0, edgecolors=None)
            # plt.scatter(list(range(0,len(dataset))), dataset["Lat"], c=dataset[column], s=20, cmap="plasma", alpha=0.5, linewidths=0, edgecolors=None)
            colorbar = plt.colorbar()
            colorbar.set_label(column)

            # Set colorbar scale
            if minmax_list:
                plt.clim(minmax_list[minmax_ind * 2], minmax_list[minmax_ind * 2 + 1])
                minmax_ind+=1


        # Finally plot
        if save:
            plt.savefig("".join(title) + "-" + column + ".png", dpi=96)
        if show:
            plt.show()

        plt.close()



def plot_map_animated(dataset, title, columns, save, show, minmax_list, min, max):
    """
        Plots each column of a Pandas DataFrame that contains the columns "Long"
        and "Lat" over a Earth map. Optionally saves the plot to disk or just
        displays the plot on the screen.

        :param dataset: pd.DataFrame
        :param title: Str. Dataset title
        :param save: Bool. Save the figure
        :param show: Bool. Show the plot
        :param minmax_list: List. Contains min and max of each graphic
        :param min: Float. Global min to set the colorbar range
        :param max: Float. Global max to set the colorbar range
        :return: None
        """
    # scales = [1, 1e9, 1e-9, 1e-9]
    # min, max = min[columns], max[columns]
    minmax_ind = 0

    for column in columns:
        data = dataset[column]
        # Set world axes
        fig = plt.figure(figsize=(1024 / 96, 768 / 96), dpi=96)
        ax = plt.axes(projection=ccrs.PlateCarree())
        # ax.coastlines()
        ax.stock_img()
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        # Plot certain places as black triangles
        plt.scatter(-76.704, -11.739, c="k", marker="^", s=4)  # Jicamarca, Peru
        plt.scatter(-71.488, 42.623, c="k", marker="^", s=4)  # MIT Haystack, EEUU
        plt.scatter(-66.752, 18.344, c="k", marker="^", s=4)  # Arecibo, Puerto Rico
        plt.scatter(-94.829, 74.697, c="k", marker="^", s=4)  # Resolute Bay, Canada
        plt.scatter(-147.488, 65.125, c="k", marker="^", s=4)  # Poker Flat, Alaska

        plt.title(column + "\n" + "".join(title))

        ims = []

        for i in range(len(dataset)):
            row = dataset.iloc[0:i + 1]
            data = row[column]

            # Set colorbar scale
            if minmax_list:
                plt.clim(minmax_list[minmax_ind * 2], minmax_list[minmax_ind * 2 + 1])

            # Plot data as scatter
            ims.append((plt.scatter(row["Lon"], row["Lat"], c=data, s=17, cmap="plasma", alpha=0.5),))

        colorbar = plt.colorbar()
        colorbar.set_label(column)

        """# Set colorbar scale
        if minmax_list:
            plt.clim(minmax_list[minmax_ind * 2], minmax_list[minmax_ind * 2 + 1])"""

        im_ani = animation.ArtistAnimation(fig, ims, interval=16, blit=True, repeat=False)

        # Set colorbar scale
        if minmax_list:
            plt.clim(minmax_list[minmax_ind * 2], minmax_list[minmax_ind * 2 + 1])

        # Finally plot
        if save:
            #plt.savefig("".join(title) + "-" + column + ".png", dpi=96)
            im_ani.save("".join(title) + "-" + column + ".mp4")

        plt.show()
        plt.close()

        minmax_ind += 1

def save_as_csv(dataset, file_name):
    dataset.to_csv(file_name, sep='\t')

def read_datafile(filename):
    """
    Reads a langmuir csv file with the format:

    time,header,Sweep voltage,Plasma voltage,Plasma temperature,Particles counter
    2018-07-19 07:03:48,0x43434301,4.017525,4.1494875,291.78375,2               # Calibration
    2018-07-19 07:03:48,0x43434302,4.0126375,3.4701250000000003,291.78375,0     # Calibration
    2018-07-19 07:03:48,0x43434303,4.022412500000001,2.9080625,291.78375,0      # Calibration
    2018-07-19 07:03:48,0x43434304,4.0126375,2.6832375,291.78375,0              # Calibration
    2018-07-19 07:05:01,0x43434305,4.0126375,2.8445250000000004,291.78375,2
    2018-07-19 07:06:01,0x43434305,4.022412500000001,3.0253625000000004,290.80625000000003,2
    ... (MORE DATA) ...
    2018-07-19 16:14:05,0x43434305,4.0028625,1.7692750000000002,296.1825,5
    2018-07-19 16:15:05,0x43434305,3.9979750000000003,1.8377000000000001,295.69375,62
    2018-07-19 16:16:05,0x43434301,4.007750000000001,4.0273,295.69375,2         # Calibration
    2018-07-19 16:16:05,0x43434302,4.0028625,3.3479375000000005,295.205,0       # Calibration
    2018-07-19 16:16:05,0x43434303,4.0126375,2.6294750000000002,295.205,0       # Calibration
    2018-07-19 16:16:05,0x43434304,4.0126375,2.09185,296.1825,0                 # Calibration

    :param filename: String File name to read
    :return: DataFrame A Pandas DataFrame with the file data
    """

    # Read CSV but skip first and last 4 rows because contains calibration data
    dataset = pd.read_csv(filename, header=0, skiprows=(1, 2, 3, 4), skipfooter=4)
    return dataset

def concatenate(files):
    """Concatenates all files in one and reads it as one langmuir csv file
    ::param files: Array of strings (names of files) to read
    :return: DataFrame A Pandas DataFrame with the file data
    """
    frames = []
    i = 0
    for filename in files:
        frames.append(read_datafile(filename))
        i += 1
    dataset = pd.concat(frames)
    return dataset

"""
MAIN FUNCTION
"""
if __name__ == "__main__":
    args = get_parameters()

    # Process each file
    if not args.concatenate:
        for filename in args.files:
            dataset = read_datafile(filename)
            tle = get_tle()
            dataset = add_long_lat(dataset, tle)
            dataset = add_plasma(dataset)
            dataset = add_is_anomaly(dataset, 600)
            if args.animate:
                plot_map_animated(dataset, filename, ["Particles counter", "Plasma current", "Electron density 300K"], args.save, True, args.color_min_max, 0, 0)
            else:
                plot_map(dataset, filename, ["Particles counter", "Plasma current", "Electron density 300K"], args.save, True,
                    args.color_min_max, 0, 0)
                dataset = make_in_max_out(dataset)
                plot_lat_in_time(dataset, 600, "Time vs lat", args.save, True)
                plot_part_in_threshold(dataset, filename, ["Particles counter"], args.save, True, args.color_min_max, 600)

    #Process all files in one
    else:
        dataset = concatenate(args.files)
        tle = get_tle()
        dataset = add_long_lat(dataset, tle)
        dataset = add_plasma(dataset)
        dataset = add_day(dataset)
        if args.animate:
            plot_map_animated(dataset, "", ["Particles counter", "Plasma current", "Electron density 300K"],
                              args.save, True, args.color_min_max, 0, 0)
        else:
            plot_map(dataset, "", ["Particles counter", "Plasma current", "Electron density 300K"], args.save,
                     True,
                     args.color_min_max, 0, 0)
            ds = pd.DataFrame(columns=dataset.columns)
            file_arr = []
            for filename in args.files:
                ds_aux = read_datafile(filename)
                ds_aux = add_long_lat(ds_aux, tle)
                ds_aux = add_plasma(ds_aux)
                ds_aux = add_is_anomaly(ds_aux, 600)
                #ds_aux = make_in_max_out(ds_aux)
                file_arr.append(ds_aux)
            ds = pd.concat(file_arr)
            ds = add_day(ds)

            #plot_lat_in_time(ds, 600, "Time vs lat", args.save, True)

            dataset = add_is_anomaly(dataset, 600)
            dataset = add_anom_cluster(dataset)
            dataset = add_season(dataset)
            save_as_csv(dataset, 'anomalies.csv')
            #plot_part_in_threshold(dataset, "", ["Particles counter"], args.save, True, args.color_min_max, 600)
            #plot_part_in_threshold(dataset, "", ["Particles counter"], args.save, True, args.color_min_max, 600)

