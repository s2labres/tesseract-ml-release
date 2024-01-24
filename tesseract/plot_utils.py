import argparse
import logging
import ujson as json
from datetime import datetime

import __main__ as main
import numpy as np
import os
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC

from tesseract import temporal, spatial

line_kwargs = {'linewidth': 1, 'markersize': 5}

force = False


# x_tick_size = 12
# y_tick_size = 14
# ax_label_size = 18
# fig_title_size = 20

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
        'text.latex.unicode': True,
        'figure.figsize': (7.2, 4.45),
        'figure.dpi': 1200,
        'savefig.dpi': 1200
    })


def get_dataset(approach):
    return {'drebin': 'drebin-parrot-v2-down',
            'mamadroid': 'mamadroidPackages-parrot-v2-down'}[approach]


def get_classifier(approach, balance=False):
    kwargs = {'class_weights': 'balanced'} if balance else {}
    if approach == 'drebin':
        return LinearSVC(**kwargs)
    if approach == 'mamadroid':
        return RandomForestClassifier(n_estimators=101, max_depth=64,
                                      n_jobs=-1, **kwargs)
    raise ValueError


def load_features(feature_set):
    fname = '../../features/{}-features'.format(feature_set)
    logging.info('Loading features...')
    with open('{}-X.json'.format(fname), 'rt') as f:
        X = json.load(f)
    [o.pop('sha256') for o in X]

    with open('{}-Y.json'.format(fname), 'rt') as f:
        y = json.load(f)
    y = [o[0] for o in y]

    with open('{}-meta.json'.format(fname), 'rt') as f:
        t = json.load(f)
    t = [o['dex_date'] for o in t]
    t = [datetime.strptime(o, '%Y-%m-%dT%H:%M:%S') for o in t]

    return X, y, t


def load_meta(feature_set):
    logging.info('Loading meta...')
    with open('../../features/{}-features-meta.json'.format(feature_set),
              'rt') as f:
        return json.load(f)


def enforce_ratios(X, y, t):
    train, tests = temporal.time_aware_indexes(t, 0, 1, 'month', '2014')
    assert len(tests) == 36

    downsampled = None
    print('{:^6} {:^6} {:^6} {:^6}'.format('MW', 'GW', 'TOT', '%MW'))

    for period_idxs in tests:
        period_idxs = np.array(period_idxs)
        y_period = y[period_idxs]

        # IF DOWNSAMPLING
        selected_idxs = spatial.downsample_to_rate(y_period)
        selected = period_idxs[selected_idxs]

        # ELSE
        # selected = period_idxs

        labels = y[selected]
        tot = len(labels)
        p = sum(labels)
        n = tot - sum(labels)
        print('{:>6} {:>6} {:>6} {:>6.1f}%'.format(p, n, tot, 100 * p / tot))

        if downsampled is None:
            downsampled = selected
        else:
            downsampled = np.hstack((downsampled, selected))

    labels = y[downsampled]
    tot = len(labels)
    p = sum(labels)
    n = tot - sum(labels)
    print('Overall')
    print('{:>6} {:>6} {:>6} {:>6.1f}%'.format(p, n, tot, 100 * p / tot))

    return downsampled


def vectorize(X, y, t):
    """Transform input data into appropriate forms for an sklearn classifier.

    Args:
        X (list): A list of dictionaries of input features for each sample.
        y (list): A list of ground truths for the data.
        t (list): A list of datetimes for the data.

    """
    logging.info('Vectorizing features...')
    vec = DictVectorizer()
    X = vec.fit_transform(X)
    y = np.asarray(y)
    t = np.asarray(t)
    return X, y, t


def style_axes(axes, periods=10):
    for i, ax in enumerate(axes):
        # Labels
        ax.set_xlabel('Testing period (month)')  # , fontsize=ax_label_size)
        # ax.set_ylabel('Score')  # , fontsize=ax_label_size)
        ax.set_ylabel('')

        # Ticks
        ax.set_xticks(range(1, periods + 1))
        ax.set_yticks(np.arange(0, 1.1, 0.1))

        labels = [str(x + 1) if x % 3 == 0 else '' for x in range(periods + 1)]
        ax.set_xticklabels(labels)

        ax.tick_params(axis='x', which='major')  # , labelsize=x_tick_size)
        ax.tick_params(axis='y', which='major')  # , labelsize=y_tick_size)

        ax.yaxis.grid(b=True, which='major', color='lightgrey', linestyle='-')

        # Axe limits
        ax.set_xlim(0, periods)
        ax.set_ylim(0, 1)

        sns.despine(ax=ax, top=True, right=True, bottom=False, left=False)


def plot_f1(ax, data, alpha=1, neg=False, label=None, color='dodgerblue',
            marker='o'):
    if label is None:
        label = 'F1 (gw)' if neg else 'F1 (mw)'
    color = '#BCDEFE' if neg else color
    series = data['f1_n'] if neg else data['f1']
    ax.plot(data.index, series, label=label, alpha=alpha, marker=marker,
            c=color, markeredgewidth=1, **line_kwargs)


def plot_roc(ax, data, alpha=1, label=None, color='dodgerblue',
             marker='o'):
    if label is None:
        label = 'AUC ROC'
    series = data['auc_roc']
    ax.plot(data.index, series, label=label, alpha=alpha, marker=marker,
            c=color, markeredgewidth=1, **line_kwargs)


def plot_f1_col(ax, data, alpha=1, neg=False, label=None, color='dodgerblue',
                marker='o'):
    if label is None:
        label = 'F1 (gw)' if neg else 'F1 (mw)'
    series = data['f1_n'] if neg else data['f1']
    ax.plot(data.index, series, label=label, alpha=alpha, marker=marker,
            c=color, markeredgewidth=1, **line_kwargs)


def plot_recall(ax, data, alpha=1, neg=False, color='red', marker='^'):
    color = '#FDB2B3' if neg else color
    label = 'Recall (gw)' if neg else 'Recall (mw)'
    series = data['recall_n'] if neg else data['recall']
    ax.plot(data.index, series, label=label, alpha=alpha,
            marker=marker, c=color, markeredgewidth=1, **line_kwargs)


def plot_precision(ax, data, alpha=1, neg=False, color='orange', marker='s'):
    color = '#FEE2B5' if neg else color
    label = 'Precision (gw)' if neg else 'Precision (mw)'
    series = data['precision_n'] if neg else data['precision']
    ax.plot(data.index, series, label=label, alpha=alpha,
            marker=marker, c=color, markeredgewidth=1, **line_kwargs)


def fill_under_f1(ax, data, alpha=1, neg=False):
    label = 'F1 (gw)' if neg else 'F1 (mw)'
    series = data['f1_n'] if neg else data['f1']
    ax.fill_between(data.index, series,
                    label='AUT({}, 24 months)'.format(label),
                    alpha=alpha, facecolor='none', hatch='//',
                    edgecolor='#BCDEFE', rasterized=True)


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


def plot_cv_mean(ax, data, alpha=1):
    ax.axhline(y=float(data), linestyle='--', linewidth=1, c='red',
               alpha=alpha, label='F1 (10-fold CV)')


def plot_x_intercept(ax, data, label='', c='limegreen', alpha=1, linewidth=1):
    ax.axvline(x=float(data), linestyle='--', linewidth=linewidth, c=c,
               alpha=alpha, label=label)


def plot_prf(ax, results, alpha=1, neg=False):
    plot_recall(ax, results, alpha, neg)
    plot_precision(ax, results, alpha, neg)
    plot_f1(ax, results, alpha, neg)


def add_legend(ax, loc='lower left'):
    lines = ax.get_lines()
    legend = ax.legend(frameon=True, handles=lines, loc=loc, prop={'size': 10})
    legend.get_frame().set_facecolor('#FFFFFF')
    legend.get_frame().set_linewidth(0)
    return legend


def set_title_sc(ax, text):
    text = text.replace('%', '\\%')  # Make TeX-safe
    ax.set_title('\\textsf{{\\textsc{{{}}}}}'.format(text))


def plotname():
    return os.path.splitext(os.path.basename(main.__file__))[0]


def save_images(plt, plot_name=None):
    plt.tight_layout()
    plot_name = plotname() if plot_name is None else plot_name
    plt.savefig('./images/png/{}.png'.format(plot_name))
    plt.savefig('./images/pdf/{}.pdf'.format(plot_name))
    plt.savefig('./images/eps/{}.eps'.format(plot_name))


def parse_args():
    global force
    p = argparse.ArgumentParser()
    p.add_argument('-f', '--force', action='store_true', help='Rerun all data')
    args = p.parse_args()
    force = args.force
    return args


parse_args()
