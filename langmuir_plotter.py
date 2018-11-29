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
