#can draw accuracy and loss graphs by using the saved log in list

def loss_plot(train_loss_list, test_loss_list, i):
    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.plot(test_loss_list, label="validation loss")
    plt.plot(train_loss_list, label="train loss")
    plt.xlabel("Epochs")
    plt.ylabel("CrossEntropyLoss")
    plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
    # plt.savefig("/home/IIS/Desktop/seunghyeon/Research/tables/noise_log/DPSGD_nm_0.1_lr_0.15_nc_1.0/{}_loss.png".format(i))
    plt.savefig("/home/IIS/Desktop/seunghyeon/Research/tables/noise_log/DPSGD_nm_0.1_lr_0.15_nc_1.0/4_loss.png")
    # plt.show()

def accuracy_plot(train_acc_list, test_acc_list, i):
    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation accuracy")
    plt.plot(test_acc_list, label="validation accuracy")
    plt.plot(train_acc_list, label="training accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("CrossEntropyLoss")
    plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
    # plt.savefig("/home/IIS/Desktop/seunghyeon/Research/tables/noise_log/DPSGD_nm_0.1_lr_0.15_nc_1.0/{}_accuracy.png".format(i))
    plt.savefig("/home/IIS/Desktop/seunghyeon/Research/tables/noise_log/DPSGD_nm_0.1_lr_0.15_nc_1.0/4_accuracy.png")
    # plt.show()
