
import os
import torch
import torch.nn
import yaml
import seaborn as sns

from tqdm import tqdm
from matplotlib import pyplot as plt
from training.dataset import loaders
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from networks import drnetq
def main(cfg):
    paths = cfg['paths']
    """
    Paths
    """
    test_dir = paths['test_imgdir']
    base_path = '/home/alonso/Documents/fundus_suitable/logs/classification/classification_2023-02-22_00_09_28/'
    model_file = os.path.join(base_path, 'checkpoints/weights.pth')

    weights = torch.load(model_file, map_location='cuda')
    model = drnetq.DRNetQ(n_classes=2).to('cuda')
    _ , test_loader = loaders(test_dir, test_dir, 256, 1)

    loop = tqdm(test_loader, ncols=120)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.eval()
    trues = []
    preds = []
    probas = []
    for batch_idx, (x, y) in enumerate(loop):
        x = x['image'].type(torch.float).to('cuda')
        y = y.type(torch.long).to('cuda')
        y_probas = model(x)
        y_pred = torch.softmax(y_probas, dim=1)
        y_pred = torch.argmax(y_pred)
        y_probas = torch.max(y_probas)
        probas.append(y_probas.detach().cpu().numpy())
        trues.append(y.detach().cpu().numpy())
        preds.append(y_pred.detach().cpu().numpy())

    confusion_mtx = confusion_matrix(trues, preds)
    sns.set(style="white")
    plt.figure()
    sns.heatmap(confusion_mtx, annot=True, fmt='',annot_kws={"size": 14})
    plt.title('Confusion Matrix')
    plt.legend(loc='best')
    plt.savefig(base_path + 'Model_CM' + '.png')

    fpr_keras, tpr_keras, thresholds_keras = roc_curve(trues, probas)
    auc_keras = auc(fpr_keras, tpr_keras)
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='AUC = {:.3f}'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(base_path + 'ROC-AUC' + '.png')

if __name__ == '__main__':
  with open('/home/alonso/Documents/fundus_suitable/configs/classifier.yaml', "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
  main(cfg)
  