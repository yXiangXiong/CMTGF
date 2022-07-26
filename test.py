import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
from sklearn.metrics import roc_auc_score

def calculate_precision(preds, reals):
    length = len(preds)
    TP = 0
    FP = 0
    for i in range(length):
        if preds[i] == 1 and reals[i] == 1:
            TP += 1
        elif preds[i] == 1 and reals[i] == 0:
            FP += 1
    precision = float(TP) / (float(FP) + float(TP))
    return precision

def calculate_f1(precision, recall):
    return 2 * (float(precision * recall) / (precision + recall))

def calculate_sensitivity(preds, reals):
    length = len(preds)
    TP = 0
    FN = 0
    for i in range(length):
        if preds[i] == 1 and reals[i] == 1:
            TP += 1
        elif preds[i] == 0 and reals[i] == 1:
            FN += 1
    sensitivity = float(TP) / (float(TP) + float(FN))
    return sensitivity

def calculate_specificity(preds, reals):
    length = len(preds)
    TN = 0
    FP = 0
    for i in range(length):
        if preds[i] == 0 and reals[i] == 0:
            TN += 1
        elif preds[i] == 1 and reals[i] == 0:
            FP += 1
    specificity = float(TN) / (float(FP) + float(TN))
    return specificity

def calculate_acc(preds, reals):
    length = len(preds)
    sum = 0
    for i in range(length):
        if preds[i] == reals[i]:
            sum += 1
    return float(sum) / length

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test
    predicted_label_prob = []
    predicted_label_index = []
    true_label = []
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, predicted_label_prob, predicted_label_index, true_label)
    acc = calculate_acc(predicted_label_index, true_label)
    sen = calculate_sensitivity(predicted_label_index, true_label)
    spe = calculate_specificity(predicted_label_index, true_label)
    prec = calculate_precision(predicted_label_index, true_label)
    f1 = calculate_f1(prec, sen)
    auc = roc_auc_score(true_label, predicted_label_prob)
    print('acc = %.3f' % acc)
    print('sensitivity = %.3f' % sen)
    print('specificity = %.3f' % spe)
    print('precision = %.3f' % prec)
    print('F1_score = %.3f' % f1)
    print('auc_score = %.3f' % auc) # 用小数算
    webpage.save()
