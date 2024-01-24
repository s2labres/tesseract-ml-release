import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict

# TODO | Remove pandas dependency

line_kwargs = {'linewidth': 1, 'markersize': 3}


# x_tick_size = 12
# y_tick_size = 14
# ax_label_size = 18
# fig_title_size = 20

def plot_decay(results, fill=True, titles=None, means=None, reject=False):
    # ------------------------------------------ #
    #  Plotting prologue                         #
    # ------------------------------------------ #

    results = [results] if isinstance(results, dict) else results
    titles = titles if titles else [''] * len(results)
    means = means if means else [None] * len(results)

    # FIXME | This is all a bit of a naff hack from before the redesign,
    # FIXME | when there was a dependency on Pandas, remove as soon as possible

    for i in range(len(results)):
        # del results[i]['auc_roc']  # Otherwise hampers the DataFrame conversion
        print(len(results[i]['f1']))
        # results[i]['f1_b'], results[i]['f1_r'], results[i]['reject_total_perc'] = [], [], []
        # for j in range(len(results[i]['transcend'])):
        #     results[i]['f1_b'].append(results[i]['transcend'][j]['f1_b'])
        #     results[i]['f1_r'].append(results[i]['transcend'][j]['f1_r'])
        #     results[i]['reject_total_perc'].append(results[i]['transcend'][j]['reject_total_perc'])
        # del results[i]['transcend']
        results[i] = pd.DataFrame(dict(results[i]),
                                  index=range(1, len(results[i]['f1']) + 1))

    # End of naffness

    set_style()
    fig, axes = plt.subplots(1, len(results))

    axes = axes if hasattr(axes, '__iter__') else (axes,)

    # ------------------------------------------ #
    #  Subplots                                  #
    # ------------------------------------------ #

    for res, ax, title, mean in zip(results, axes, titles, means):
        # plot_prf(ax, res, 0.3, neg=True)
        plot_prf(ax, res)
        if mean is not None:
            plot_cv_mean(ax, mean)
        if fill:
            fill_under_f1(ax, res)
        if reject:
            plot_baseline_f1(ax, res)
            plot_rej_f1(ax, res)
            plot_rejected(ax, res)
        ax.set_title(title)

    # Legend
    add_legend(axes[0])

    # ------------------------------------------ #
    #  Plotting epilogue                         #
    # ------------------------------------------ #

    style_axes(axes, len(results[0]['f1']))
    fig.set_size_inches(6 * len(results), 4)
    plt.tight_layout()

    return plt


def plot_decay1(results, fill=True, titles=None, means=None, reject=False):
    # ------------------------------------------ #
    #  Plotting prologue                         #
    # ------------------------------------------ #

    results = [results] if isinstance(results, dict) else results
    titles = titles if titles else [''] * len(results)
    means = means if means else [None] * len(results)

    # FIXME | This is all a bit of a naff hack from before the redesign,
    # FIXME | when there was a dependency on Pandas, remove as soon as possible
    data = defaultdict(lambda: [])
    for result in results:
        for i in result:
            data[i].append(result[i])
    results = [pd.DataFrame(dict(data), index=range(1, len(data['f1_b']) + 1))]

    # End of naffness

    set_style()
    fig, axes = plt.subplots(1, len(results))

    axes = axes if hasattr(axes, '__iter__') else (axes,)

    # ------------------------------------------ #
    #  Subplots                                  #
    # ------------------------------------------ #

    for res, ax, title, mean in zip(results, axes, titles, means):
        # plot_prf(ax, res, 0.3, neg=True)
        plot_prf(ax, res)
        if mean is not None:
            plot_cv_mean(ax, mean)
        if fill:
            fill_under_f1(ax, res)
        if reject:
            plot_baseline_f1(ax, res)
            plot_rej_f1(ax, res)
            plot_rejected(ax, res)
        ax.set_title(title)

    # Legend
    add_legend(axes[0])

    # ------------------------------------------ #
    #  Plotting epilogue                         #
    # ------------------------------------------ #

    style_axes(axes, len(results[0]['f1_b']))
    fig.set_size_inches(6 * len(results), 4)
    plt.tight_layout()

    return plt


def set_style():
    sns.set_context('paper')
    sns.set(font='serif')

    sns.set('paper', font='serif', style='ticks', rc={
        'font.family': 'serif',
        'legend.fontsize': 'medium',
        'xtick.labelsize': 'medium',
        'ytick.labelsize': 'medium',
        'axes.labelsize': 'x-large',
        'axes.titlesize': 'x-large',
        'axes.labelpad': 6.0,
        'figure.titlesize': 'x-large',
        'text.usetex': True,
        'figure.figsize': (3.6, 4.45),
        'figure.dpi': 1200,
        'savefig.dpi': 1200
    })


def style_axes(axes, periods, granularity='Month'):
    for i, ax in enumerate(axes):
        # Labels
        ax.set_xlabel(f'Testing period ({granularity})')  # , fontsize=ax_label_size)
        # ax.set_ylabel('Score')  # , fontsize=ax_label_size)
        ax.set_ylabel('')

        # Ticks
        ax.set_xticks(range(1, periods + 1))
        ax.set_yticks(np.arange(0, 1.1, 0.1))

        if periods > 12:
            labels = [str(x + 1) if x % 3 == 0
                      else '' for x in range(periods)]
        else:
            labels = [str(x + 1) for x in range(periods)]

        ax.set_xticklabels(labels)

        ax.tick_params(axis='x', which='major')  # , labelsize=x_tick_size)
        ax.tick_params(axis='y', which='major')  # , labelsize=y_tick_size)

        ax.yaxis.grid(visible=True, which='major', color='lightgrey', linestyle='-')

        # Axe limits
        ax.set_xlim(0.8, periods)
        ax.set_ylim(0, 1)

        sns.despine(ax=ax, top=True, right=True, bottom=False, left=False)


def plot_baseline_f1(ax, data, alpha=1.0, color='gray', linestyle='--'):
    label = 'F1 (no rejection)'
    series = data['f1_b']
    ax.plot(data.index + 1, series, label=label, alpha=alpha, linestyle=linestyle,
            c=color, markeredgewidth=1, **line_kwargs)


def plot_rej_f1(ax, data, alpha=1.0, color='red', marker='o'):
    label = 'F1 (rejection)'
    series = data['f1_r']
    ax.plot(data.index + 1, series, label=label, alpha=alpha, marker=marker,
            c=color, markeredgewidth=1, **line_kwargs)


def plot_rejected(ax, data, alpha=0.6, color='#C0C0C0'):
    series = data['reject_total_perc']
    ax.bar(data.index + 1, series, width=0.7, color=color, alpha=alpha)


def plot_f1(ax, data, alpha=1.0, neg=False, label=None, color='dodgerblue',
            marker='o'):
    if label is None:
        label = 'F1 (gw)' if neg else 'F1 (mw)'

    if neg:
        if color=='dodgerblue':
            color = '#BCDEFE'

    series = data['f1_n'] if neg else data['f1']
    ax.plot(data.index + 1, series, label=label, alpha=alpha, marker=marker,
            c=color, markeredgewidth=1, **line_kwargs)


def plot_recall(ax, data, alpha=1.0, neg=False, color='red', marker='^'):
    label = 'Recall (gw)' if neg else 'Recall (mw)'
    color = '#FDB2B3' if neg else color
    series = data['recall_n'] if neg else data['recall']
    ax.plot(data.index + 1, series, label=label, alpha=alpha,
            marker=marker, c=color, markeredgewidth=1, **line_kwargs)


def plot_precision(ax, data, alpha=1.0, neg=False, color='orange', marker='s'):
    label = 'Precision (gw)' if neg else 'Precision (mw)'
    color = '#FEE2B5' if neg else color
    series = data['precision_n'] if neg else data['precision']
    ax.plot(data.index + 1, series, label=label, alpha=alpha,
            marker=marker, c=color, markeredgewidth=1, **line_kwargs)


def fill_under_f1(ax, data, alpha=1, neg=False):
    label = 'F1 (gw)' if neg else 'F1 (mw)'
    series = data['f1_n'] if neg else data['f1']
    ax.fill_between(data.index + 1, series,
                    label='AUT({}, 24 months)'.format(label),
                    alpha=alpha, facecolor='none', hatch='//',
                    edgecolor='#BCDEFE', rasterized=True)


def plot_cv_mean(ax, data, alpha=1):
    ax.axhline(y=float(data), linestyle='--', linewidth=1, c='red',
               alpha=alpha, label='F1 (10-fold CV)')


def plot_origin(ax, data, alpha=1):
    ax.axhline(y=float(data), linestyle='-.', linewidth=1, c='black',
               alpha=alpha, label='F1 (original paper)')


def plot_prf(ax, results, alpha=1.0, neg=False):
    plot_f1(ax, results, alpha, neg)
    plot_recall(ax, results, alpha, neg)
    plot_precision(ax, results, alpha, neg)



def add_legend(ax, loc='lower left'):
    lines = ax.get_lines()
    legend = ax.legend(frameon=True, handles=lines, loc=loc, prop={'size': 8},  # Reduced font size
                       borderpad=0.5,  # Padding inside the legend box
                       labelspacing=0.5,  # Vertical spacing between legend items
                       handlelength=1,  # Length of the legend handles
                       handletextpad=0.5)  # Spacing between handle and text
    legend.get_frame().set_facecolor('#FFFFFF')
    legend.get_frame().set_linewidth(0)
    return legend


def save_images(plt, path, plot_name):
    plt.tight_layout()
    plt.savefig(os.path.join(path, './png/{}.png'.format(plot_name)))
    plt.savefig(os.path.join(path, './pdf/{}.pdf'.format(plot_name)))
    plt.savefig(os.path.join(path, './eps/{}.eps'.format(plot_name)))


def plot_old_f1(ax, data, alpha=1, neg=False, label=None,
                color='#C0C0C0', marker=''):
    if label is None:
        label = 'F1 (gw)' if neg else 'F1 (mw)'
    series = data['f1_n'] if neg else data['f1']
    ax.plot(data.index, series, label=label, alpha=alpha, linestyle='--',
            marker=marker, c=color, markeredgewidth=1, linewidth=2)


def plot_old_metric(ax, data, metric, alpha=1, neg=False, label=None,
                    color='#C0C0C0', marker=''):
    if label is None:
        label = metric + ' (gw)' if neg else metric + ' (mw)'
        label = label.title()
    series = data[metric + '_n'] if neg else data[metric]
    ax.plot(data.index, series, label=label, alpha=alpha, linestyle='--',
            marker=marker, c=color, markeredgewidth=1, linewidth=2)


def set_title_sc(ax, text):
    text = text.replace('%', '\\%')  # Make TeX-safe
    ax.set_title('\\textsf{{\\textsc{{{}}}}}'.format(text))


def main():
    import pickle as pkl

    results = pkl.load(
        open('/Users/mark/Documents/Git/transcend-release/timeseries_cred_conf/ice_p_val_results.p', 'rb'))
    plot = plot_decay1(results, reject=True, titles=['ICE default'])
    plot.savefig("/Users/mark/Desktop/Tesseract-journal/ICE.pdf")


if __name__ == '__main__':
    main()
