import numpy as np
import matplotlib.pyplot as plt


def load_data(path):
    with open('plots/text_files/'+path, 'r') as f:
        lines = f.read().splitlines()
        rew, batch = [], []
        for i, l in enumerate(lines):
            r = float(l)
            batch.append(1+i)
            rew.append(r)
    return batch, rew


def plot(path, label):
    b,r = load_data(path)
    plt.plot(b,r, label = label)






plot("test_loss_neural_regression.txt", label = 'Regressor')
plot("test_loss_neural_clf_text.txt", label = 'Classifier')
plt.xlabel("epoch")
plt.ylabel("loss (MAE)")
plt.title("Test loss - classifier and regressor ")
plt.legend()
plt.savefig("plots/images/test_loss_clfvsreg")
plt.clf()

plot("test_loss_neural_clf_no_text.txt", label = 'No text features')
plot("test_loss_neural_clf_text.txt", label = 'Text features')
plt.xlabel("epoch")
plt.ylabel("loss (MAE)")
plt.title("Test loss - with and without text features ")
plt.legend()
plt.savefig("plots/images/test_loss_textvsnotext")
plt.clf()


plot("train_loss_neural_clf_no_text.txt", label = 'No text features')
plot("train_loss_neural_clf_text.txt", label = 'Text features')
plt.xlabel("epoch")
plt.ylabel("loss (Cross Entropy)")
plt.title("Train loss - with and without text features ")
plt.legend()
plt.savefig("plots/images/train_loss_textvsnotext")
plt.clf()



depths = [3,4,5,6,7,8]
xgb_text_train = [138.31, 134.35, 128.46, 120.52, 110.61, 100.96]
xgb_text_test = [141.23, 140.51, 140.02, 140.63, 140.22, 140.94]

xgb_notext_train = [139.97, 137.27, 135.73, 133.51, 131.15, 129.37]  
xgb_notext_test = [139.78, 139.52, 139.23, 139.33, 139.61, 140.04]

plt.plot(depths, xgb_text_train, label = 'text')
plt.plot(depths, xgb_notext_train, label = 'no text')
plt.xlabel('Depth')
plt.ylabel('loss (MAE)')
plt.title("XGB models - training loss")
plt.legend()
plt.savefig("plots/images/xgb_train_loss_textvsnotext")
plt.clf()

plt.plot(depths, xgb_text_test, label = 'text')
plt.plot(depths, xgb_notext_test, label = 'no text')
plt.xlabel('Depth')
plt.ylabel('loss (MAE)')
plt.title("XGB models - test loss")
plt.legend()
plt.savefig("plots/images/xgb_test_loss_textvsnotext")
plt.clf()


_, train_loss_rf = load_data("train_loss_rf_notext.txt")
_, test_loss_rf = load_data("test_loss_rf_notext.txt")
depths = [5+i for i in range(34)]


plt.plot(depths, train_loss_rf, label = 'training loss')
# plt.plot(depths, test_loss_rf, label = 'test loss')
plt.xlabel("depth")
plt.ylabel("loss (MAE)")
plt.title("random forest model - training loss")
plt.legend()
plt.savefig("plots/images/rf_train_loss")
plt.clf()

plt.plot(depths, test_loss_rf, label = 'test loss')
plt.xlabel("depth")
plt.ylabel("loss (MAE)")
plt.title("random forest model - test loss")
plt.legend()
plt.savefig("plots/images/rf_test_loss")
plt.clf()

