import tempfile
import os
import subprocess

import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt


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


def evaluate_ft_args(train_loc, test_loc, model_loc, output_loc, args):
    train_fasttext(train_loc, model_loc, args)
    predictions = predict_fasttext(test_loc, model_loc)
    pred_targets = [news.target_names.index(cls) for cls in predictions]
    evaluate_model(output_loc, test_docs['target'].tolist(), pred_targets, news.target_names)
    write_args_file(output_loc, args)


tmp_dir = tempfile.TemporaryDirectory().name
config = {
    'output_dir': 'output',
    'tmp_dir': tmp_dir,
    'train_data_loc': tmp_dir + '/train_data',
    'test_data_loc': tmp_dir + '/test_data',
    'test_pred_loc': tmp_dir + '/test_pred',
    'model_loc': tmp_dir + '/ftclf_model'
}

ft_args = {
    'lr': 0.05,
    'lrUpdateRate': 100,
    'dim': 300,
    'ws': 5,
    'epoch': 50,
    'neg': 5,
    'loss': 'softmax',
    'thread': 3,
    #'pretrainedVectors': '/home/dorian/Downloads/wiki-news-300d-1M.vec'
}


if not os.path.exists(config['tmp_dir']):
    os.makedirs(config['tmp_dir'])

print("Running with config:")
for k, v in config.items():
    print('>', k, ":", v)

if __name__ == "__main__":

    # Load data
    news = fetch_20newsgroups()
    all_docs = pd.DataFrame({
        'data': news.data,
        'target': news.target,
        'label': [news.target_names[i] for i in news.target]
    })

    all_docs['cleaned_text'] = cleanup_text(all_docs['data'])
    all_docs = all_docs.assign(ft_format=lambda x: '__label__' + x.label + ' ' + x.cleaned_text)
    train_docs, test_docs = train_test_split(all_docs, test_size=0.2)

    write_to_disk(train_docs['ft_format'].tolist(), config['train_data_loc'])
    write_to_disk(test_docs['ft_format'].tolist(), config['test_data_loc'])

    evaluate_ft_args(train_loc=config['train_data_loc'],
                     test_loc=config['test_data_loc'],
                     model_loc=config['model_loc'],
                     output_loc=config['output_dir'],
                     args=ft_args)

    # Todo: Gridsearch
    # Todo: K-fold cross validation