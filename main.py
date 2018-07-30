import tempfile
import os
import subprocess
import shutil
from itertools import product

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


def print_confusion_matrix(conf_matrix, class_names):
    df_cm = pd.DataFrame(
        conf_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure()
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig


def evaluate_model(output_loc, actual, predicted, target_names):
    clf_report = classification_report(actual, predicted, target_names=target_names)
    print("Saving classification report to: {}".format(output_loc + '/clf_report'))
    with open(output_loc + '/clf_report', 'a+') as f:
        f.write(clf_report)

    cnf_matrix = confusion_matrix(actual, predicted)
    fig = print_confusion_matrix(cnf_matrix, class_names=target_names)
    print("Saving confusion matrix to: {}".format(output_loc + '/cnf_matrix.png'))
    fig.savefig(output_loc + '/cnf_matrix.png')


def write_to_disk(docs, write_loc):
    print("Writing training data to: {}".format(write_loc))
    with open(write_loc, 'a+') as f:
        for doc in docs:
            f.write(doc)
            f.write('\n')


def cleanup_text(text_col):
    return text_col.replace(r'\n', ' ', regex=True)


def write_args_file(output_loc, args):
    with open(output_loc + '/model_args', 'a+') as f:
        for _k, _v in args.items():
            f.write(_k + ' : ' + str(_v))
            f.write('\n')


def train_fasttext(data_loc, model_loc, args):
    args_list = ['fasttext', 'supervised', '-input', data_loc, '-output', model_loc]
    for _k, _v in args.items():
        args_list.append('-' + _k)
        args_list.append(str(_v))

    print("Training classifier with command: {0}".format(" ".join(args_list)))
    subprocess.run(args_list)
    print("Classifier successfully trained!")


def predict_fasttext(data_loc, model_loc):
    args_list = ['fasttext', 'predict', model_loc + '.bin', data_loc]
    print("Predicting classes with command: {0}".format(" ".join(args_list)))
    result = subprocess.run(args_list, stdout=subprocess.PIPE)
    preds = result.stdout.decode("utf-8").replace('__label__', '').split('\n')
    print("Classes successfully predicted")

    # Last row is an empty string (stdout)
    return preds[:-1]


def evaluate_ft_args(train_data, test_data, class_names, args, output_dir):
    tmp_dir = tempfile.TemporaryDirectory().name
    config = {
        'tmp_dir': tmp_dir,
        'train_data_loc': tmp_dir + '/train_data',
        'test_data_loc': tmp_dir + '/test_data',
        'test_pred_loc': tmp_dir + '/test_pred',
        'model_loc': tmp_dir + '/ftclf_model'
    }
    if not os.path.exists(config['tmp_dir']):
        os.makedirs(config['tmp_dir'])

    write_to_disk(train_data['ft_format'].tolist(), config['train_data_loc'])
    write_to_disk(test_data['ft_format'].tolist(), config['test_data_loc'])

    train_fasttext(config['train_data_loc'], config['model_loc'], args)
    predictions = predict_fasttext(config['test_data_loc'], config['model_loc'])

    act_index = [news.target_names.index(cls) for cls in test_data['label']]
    pred_index = [news.target_names.index(cls) for cls in predictions]
    evaluate_model(output_dir, act_index, pred_index, class_names)
    acc = accuracy_score(act_index, pred_index)

    shutil.rmtree(tmp_dir)
    return acc


def expand_grid(dct):
    return [dict(zip(dct.keys(), arg_set)) for arg_set in product(*dct.values())]


args_grid = {
    'lr': [0.05],
    'lrUpdateRate': [100],
    'dim': [300],
    'ws': [5],
    'epoch': [25],
    'neg': [5],
    'loss': ['ns'],
    'thread': [3],
    # 'pretrainedVectors': '/home/dorian/Downloads/wiki-news-300d-1M.vec'
}

if __name__ == "__main__":
    # Load data
    news = fetch_20newsgroups()
    all_docs = pd.DataFrame({
        'data': news.data,
        'label': [news.target_names[i] for i in news.target]
    })

    all_docs['data'] = cleanup_text(all_docs['data'])
    all_docs = all_docs.assign(ft_format=lambda x: '__label__' + x.label + ' ' + x.data)

    le = LabelEncoder()
    le.fit(all_docs['label'])
    target_names = list(le.classes_)

    # kfold split
    skf = StratifiedKFold(n_splits=2)

    args_grid_list = expand_grid(args_grid)

    for n in range(len(args_grid_list)):
        print("GridSearch progress: {}/{}".format(n, len(args_grid_list)))
        args = args_grid_list[n]

        num_split = -1
        acc_arr = np.array([])
        for train_idx, test_idx in skf.split(all_docs['data'], all_docs['label']):
            num_split += 1
            train_df = all_docs.iloc[train_idx]
            test_df = all_docs.iloc[test_idx]

            output_dir = 'output/{run}/{split}'.format(run=n, split=num_split)
            os.makedirs(output_dir)
            acc = evaluate_ft_args(train_data=train_df,
                                   test_data=test_df,
                                   class_names=target_names,
                                   args=args,
                                   output_dir=output_dir)
            acc_arr = np.append(acc_arr, acc)

        write_args_file("output/{}".format(n), args)
        with open("output/{}/mean_acc:{}".format(n, round(acc_arr.mean(), 3)), 'a+') as f:
            f.write(' ')
        with open("output/{}/std_acc:{}".format(n, round(acc_arr.std(), 3)), 'a+') as f:
            f.write(' ')

    # TODO: Add output of whole script (best args, with mean+std of acc)
