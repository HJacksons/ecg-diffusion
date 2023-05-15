from matplotlib.font_manager import FontProperties
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import pearsonr
import seaborn as sns
import pandas as pd
import numpy as np
import helpers

DATA = 'unet'  # unet, diffwave, p2p


def get_statistics(data):
    _, test_dataloader = helpers.get_dataloader(target='test', batch_size=1, shuffle=False)

    rr_list = []
    qrs_list = []
    qt_list = []
    v_rate_list = []
    r_peak_i_list = []
    r_peak_v1_list = []

    generated_rr_list = []
    generated_qrs_list = []
    generated_qt_list = []
    generated_v_rate_list = []
    generated_r_peak_i_list = []
    generated_r_peak_v1_list = []

    for _, (_, features) in enumerate(test_dataloader, 0):
        rr = int(features[:, 0].squeeze().cpu().numpy())
        qrs = int(features[:, 1].squeeze().cpu().numpy())
        qt = int(features[:, 3].squeeze().cpu().numpy())
        v_rate = int(features[:, 4].squeeze().cpu().numpy())
        r_peak_i = int(features[:, 5].squeeze().cpu().numpy())
        r_peak_v1 = int(features[:, 7].squeeze().cpu().numpy())

        rr_list.append(rr)
        qrs_list.append(qrs)
        qt_list.append(qt)
        v_rate_list.append(v_rate)
        r_peak_i_list.append(r_peak_i)
        r_peak_v1_list.append(r_peak_v1)

    for file_index in range(0, 952):
        if data == 'p2p':
            ecg_df = pd.read_csv(f"{data}/pulse2pulse-{file_index}.csv")
        else:
            ecg_df = pd.read_csv(f"{data}/{data}-{file_index}.csv")

        rr = ecg_df.loc[0, 'rr']
        qrs = ecg_df.loc[0, 'qrs']
        qt = ecg_df.loc[0, 'qt']
        v_rate = ecg_df.loc[0, 'ventr_rate']
        r_peak_i = ecg_df.loc[0, 'r_peak_i']
        r_peak_v1 = ecg_df.loc[0, 'r_peak_v1']

        generated_rr_list.append(rr)
        generated_qrs_list.append(qrs)
        generated_qt_list.append(qt)
        generated_v_rate_list.append(v_rate)
        generated_r_peak_i_list.append(r_peak_i)
        generated_r_peak_v1_list.append(r_peak_v1)

    rr_std = np.std(rr_list)
    rr_mean = np.mean(rr_list)

    qrs_std = np.std(qrs_list)
    qrs_mean = np.mean(qrs_list)

    qt_std = np.std(qt_list)
    qt_mean = np.mean(qt_list)

    v_rate_std = np.std(v_rate_list)
    v_rate_mean = np.mean(v_rate_list)

    r_peak_i_std = np.std(r_peak_i_list)
    r_peak_i_mean = np.mean(r_peak_i_list)

    r_peak_v1_std = np.std(r_peak_v1_list)
    r_peak_v1_mean = np.mean(r_peak_v1_list)

    # print(f'rr_std: {rr_std}, rr_mean: {rr_mean}')
    # print(f'qrs_std: {qrs_std}, qrs_mean: {qrs_mean}')
    # print(f'qt_std: {qt_std}, qt_mean: {qt_mean}')
    # print(f'v_rate_std: {v_rate_std}, v_rate_mean: {v_rate_mean}')
    # print(f'r_peak_i_std: {r_peak_i_std}, r_peak_i_mean: {r_peak_i_mean}')
    # print(f'r_peak_v1_std: {r_peak_v1_std}, r_peak_v1_mean: {r_peak_v1_mean}')

    return generated_rr_list, generated_qt_list, rr_list, qt_list


def get_heart_rate_plot(ventricular_rate_list):
    sns.set_style('darkgrid')

    # v_rate_list = get_std_and_mean(DATA)
    l = [int(x) for x in ventricular_rate_list]

    left_border = 60
    right_border = 100
    main_color = mcolors.to_rgb('steelblue')
    highlight_color = mcolors.to_rgb('rosybrown')

    binvals, bins, patches = plt.hist(l, bins=50, color=main_color)

    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    for p, x in zip(patches, bin_centers):
        if x < left_border or x > right_border:
            p.set_facecolor(highlight_color)
            p.set_edgecolor(highlight_color)
        else:
            p.set_facecolor(main_color)
            p.set_edgecolor(main_color)

    font_title = FontProperties()
    font_title.set_family('sans-serif')
    font_title.set_size(17)

    font_axis_labels = FontProperties()
    font_axis_labels.set_family('sans-serif')
    font_axis_labels.set_size(14)

    plt.title('DiffWave Generated ECG Distribution', fontproperties=font_title)
    plt.xlabel('Heart Rate [bpm]', fontproperties=font_axis_labels)
    plt.ylabel('Count', fontproperties=font_axis_labels)
    plt.xticks(fontproperties=font_axis_labels)
    plt.yticks(fontproperties=font_axis_labels)
    plt.savefig(f'hr_{DATA}.pdf')


# get_heart_rate_plot(ventricular_rate_list)


def get_qt_rr_plot():
    generated_rr_list, generated_qt_list, rr_list, qt_list = get_statistics(DATA)

    sns.set_style("darkgrid")

    sns.scatterplot(x=generated_rr_list, y=generated_qt_list, color="rosybrown", label='generated', s=10)
    sns.scatterplot(x=rr_list, y=qt_list, color="steelblue", label='real', s=10)

    # compute Pearson correlation coefficient
    r_generated, p1 = pearsonr(generated_rr_list, generated_qt_list)
    slope_generated = r_generated * np.std(generated_qt_list) / np.std(generated_rr_list)
    intercept_generated = np.mean(generated_qt_list) - slope_generated * np.mean(generated_rr_list)

    r_real, p2 = pearsonr(rr_list, qt_list)
    slope_real = r_real * np.std(qt_list) / np.std(rr_list)
    intercept_real = np.mean(qt_list) - slope_real * np.mean(rr_list)

    # Add line representing distribution of data
    xlim = plt.xlim()
    ylim = plt.ylim()
    x = np.linspace(xlim[0], xlim[1], 100)

    # define line equation using slope and intercept
    y_real = slope_real * x + intercept_real
    y_generated = slope_generated * x + intercept_generated

    sns.lineplot(x=x, y=y_real, color='midnightblue', label=f'real Pearson r`2 = {pow(r_real, 2):.2f}')
    sns.lineplot(x=x, y=y_generated, color="brown", label=f'generated Pearson r`2 = {pow(r_generated, 2):.2f}')

    font_title = FontProperties()
    font_title.set_family('sans-serif')
    font_title.set_size(16)

    font_axis_labels = FontProperties()
    font_axis_labels.set_family('sans-serif')
    font_axis_labels.set_size(8)

    font_axis_labels1 = FontProperties()
    font_axis_labels1.set_family('sans-serif')
    font_axis_labels1.set_size(12)

    plt.xlabel("RR Interval (ms)", fontproperties=font_axis_labels1)
    plt.ylabel("QT Interval (ms)", fontproperties=font_axis_labels1)
    plt.title("UNet - Generated vs Real ECG Distributions", fontproperties=font_title)
    plt.xticks(fontproperties=font_axis_labels)
    plt.yticks(fontproperties=font_axis_labels)
    plt.savefig(f'qt_rr_{DATA}.pdf')


get_qt_rr_plot()
