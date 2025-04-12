import numpy as np
import os
import pickle
from PIL import Image
from My_network import My_network
import matplotlib.pyplot as plt
import tqdm
import seaborn as sns
import pandas as pd

class My_dataset:
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.data = []
        self.labels = []
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
        if self.train:
            for i in range(1,6):
                file_path = os.path.join(self.root,'cifar-10-batches-py',f'data_batch_{i}')
                with open(file_path, 'rb') as f:
                    img_dict = pickle.load(f,encoding='bytes')
                    self.data.append(img_dict[b'data'])
                    self.labels.extend(img_dict[b'labels'])
            self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
            self.data = self.data.transpose(0, 2, 3, 1)
        else:
            file_path = os.path.join(self.root,'cifar-10-batches-py',f'test_batch')
            with open(file_path, 'rb') as f:
                img_dict = pickle.load(f,encoding='bytes')
                self.data=img_dict[b'data']
                self.labels=img_dict[b'labels']
                self.data = self.data.reshape(-1, 3, 32, 32)
                self.data = self.data.transpose(0, 2, 3, 1)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]

        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, label

class MyTransform:
    def __init__(self):
        self.mean = np.array([0.4914, 0.4822, 0.4465])
        self.std = np.array([0.2470, 0.2435, 0.2616])
    def __call__(self, img):
        img = np.array(img).astype(np.float32)/255.0
        img_array = (img-self.mean)/self.std
        return img_array.reshape(-1)
def train(file_path,epoch):
    transform = MyTransform()
    train_dataset = My_dataset(root='./data', train=True, transform=transform)

    X_train=[]
    Y_train=[]
    for i in tqdm.tqdm(range(len(train_dataset))):
        x,y = train_dataset[i]
        X_train.append(x)
        Y_train.append(y)
    X_train = np.array(X_train)
    Y_train_onehot = np.zeros((len(train_dataset), 10))
    for i,y in enumerate(Y_train):
        Y_train_onehot[i,y] = 1
    X_train_without_val=X_train[0:45000]
    Y_train_without_val=Y_train_onehot[0:45000]
    X_train_val=X_train[45000:]
    Y_train_val=Y_train_onehot[45000:]
    model = My_network(
        input_size=3072,
        layer1_size=128,
        layer2_size=64,
        output_size=10,
    )
    print('start training...')
    losses,val_losses,acc_lst = model.train(X_train_without_val, Y_train_without_val,X_train_val, Y_train_val,epochs=epoch)

    plt.figure(figsize=(10, 6))
    plt.subplot(1,2,1)
    plt.plot(losses,'b-',label='train loss')
    plt.plot(val_losses,'r--',label='validation loss')
    plt.savefig('loss_curve.png')
    plt.subplot(1,2,2)
    plt.plot(acc_lst,'b-',label='val acc')
    plt.savefig('acc_curve.png')
    plt.tight_layout()
    plt.show()
    model.save_params(file_path)
def test():
    transform = MyTransform()
    test_dataset = My_dataset(root='./data', train=False, transform=transform)
    X_test,Y_test = [],[]
    for i in tqdm.tqdm(range(len(test_dataset))):
        x,y = test_dataset[i]
        X_test.append(x)
        Y_test.append(y)
    X_test = np.array(X_test)
    Y_test_onehot = np.zeros((len(test_dataset), 10))
    for i,y in enumerate(Y_test):
        Y_test_onehot[i,y] = 1
    model = My_network(
        input_size=3072,
        layer1_size=128,
        layer2_size=64,
        output_size=10,
    )
    model.load_params()
    # img = X_test[0]
    # label = Y_test[0]
    # print(model.predict(img))
    # print(test_dataset.classes[label])
    print(model.evaluate(X_test, Y_test_onehot))





def export_visualize_parameters(network, save_dir='./network_params/'):
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    sns.histplot(network.W1.flatten(), kde=True)
    plt.title('W1 weight distribution')
    plt.subplot(2, 3, 2)
    sns.histplot(network.W2.flatten(), kde=True)
    plt.title('W2 weight distribution')
    plt.subplot(2, 3, 3)
    sns.histplot(network.W3.flatten(), kde=True)
    plt.title('W3 weight distribution')
    plt.subplot(2, 3, 4)
    sns.histplot(network.b1.flatten(), kde=True)
    plt.title('b1 bias distribution')
    plt.subplot(2, 3, 5)
    sns.histplot(network.b2.flatten(), kde=True)
    plt.title('b2 bias distribution')
    plt.subplot(2, 3, 6)
    sns.histplot(network.b3.flatten(), kde=True)
    plt.title('b3 bias distribution')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/parameter_distributions.png')
    print(f"参数分布图已保存为: {save_dir}/parameter_distributions.png")

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sample_size = min(50, min(network.W1.shape))
    i_indices = np.random.choice(network.W1.shape[0], sample_size, replace=False)
    j_indices = np.random.choice(network.W1.shape[1], sample_size, replace=False)
    W1_sample = network.W1[np.ix_(i_indices, j_indices)]
    sns.heatmap(W1_sample, cmap='coolwarm')
    plt.title(f'W1 weight heat graph (sample {sample_size}x{sample_size})')

    plt.subplot(1, 3, 2)
    if network.W2.shape[0] > 50 or network.W2.shape[1] > 50:
        sample_size = min(50, min(network.W2.shape))
        i_indices = np.random.choice(network.W2.shape[0], sample_size, replace=False)
        j_indices = np.random.choice(network.W2.shape[1], sample_size, replace=False)
        W2_sample = network.W2[np.ix_(i_indices, j_indices)]
        sns.heatmap(W2_sample, cmap='coolwarm')
        plt.title(f'W2 weight heat graph (sample {sample_size}x{sample_size})')
    else:
        sns.heatmap(network.W2, cmap='coolwarm')
        plt.title('W2 weight heat graph')

    plt.subplot(1, 3, 3)
    sns.heatmap(network.W3, cmap='coolwarm')
    plt.title('W3 weight heat graph')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/weight_heatmaps.png')
    print(f"already downloaded: {save_dir}/weight_heatmaps.png")

    stats = {
        'layer': ['W1', 'W2', 'W3', 'b1', 'b2', 'b3'],
        'min': [network.W1.min(), network.W2.min(), network.W3.min(),
                network.b1.min(), network.b2.min(), network.b3.min()],
        'max': [network.W1.max(), network.W2.max(), network.W3.max(),
                network.b1.max(), network.b2.max(), network.b3.max()],
        'mean': [network.W1.mean(), network.W2.mean(), network.W3.mean(),
                 network.b1.mean(), network.b2.mean(), network.b3.mean()],
        'std': [network.W1.std(), network.W2.std(), network.W3.std(),
                network.b1.std(), network.b2.std(), network.b3.std()],
        'nonzero': [np.count_nonzero(network.W1) / network.W1.size,
                    np.count_nonzero(network.W2) / network.W2.size,
                    np.count_nonzero(network.W3) / network.W3.size,
                    np.count_nonzero(network.b1) / network.b1.size,
                    np.count_nonzero(network.b2) / network.b2.size,
                    np.count_nonzero(network.b3) / network.b3.size]
    }


    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(f'{save_dir}/parameter_stats.csv', index=False)
    print(f"already saved: {save_dir}/parameter_stats.csv")

    return {
        'W1': network.W1,
        'W2': network.W2,
        'W3': network.W3,
        'b1': network.b1,
        'b2': network.b2,
        'b3': network.b3
    }



def find_super_params(X_train, Y_train, X_val, Y_val):
    L2_lamda_list = [0.0001, 0.001, 0.00001]
    learning_rate_list = [0.001, 0.01, 0.1]
    hidden_layer_size_list = [ [128, 64, 10],[256, 128, 10] ]

    best_val_loss = float('inf')
    best_params = None
    for L2 in L2_lamda_list:
        for learning_rate in learning_rate_list:
            for hidden_layer_size in hidden_layer_size_list:
                print(f"测试参数组合: L2={L2}, lr={learning_rate}, layers={hidden_layer_size}")

                model = My_network(
                    input_size=3072,
                    layer1_size=hidden_layer_size[0],
                    layer2_size=hidden_layer_size[1],
                    output_size=hidden_layer_size[2],
                    L2_lamda=L2,
                    learning_rate=learning_rate
                )

                try:
                    losses, val_losses = model.train(X_train, Y_train, X_val, Y_val)
                    final_val_loss = val_losses[-1]

                    model.save_params(
                        filename=f'L2_{L2}_lr_{learning_rate}_layers_{hidden_layer_size[0]}_{hidden_layer_size[1]}_params.npz')
                    np.savez(
                        file=f'L2_{L2}_lr_{learning_rate}_layers_{hidden_layer_size[0]}_{hidden_layer_size[1]}_loss.npz',
                        losses=losses,
                        val_losses=val_losses
                    )

                    if final_val_loss < best_val_loss:
                        best_val_loss = final_val_loss
                        best_params = {
                            'L2': L2,
                            'learning_rate': learning_rate,
                            'hidden_layers': hidden_layer_size,
                            'val_loss': final_val_loss
                        }

                    print(
                        f"L2={L2}, lr={learning_rate}, layers={hidden_layer_size}, val_loss={final_val_loss:.4f}")
                except Exception as e:
                    print(f"错误信息: {e}")


    if best_params:
        print("\n最佳参数组合:")
        print(f"L2 正则化: {best_params['L2']}")
        print(f"学习率: {best_params['learning_rate']}")
        print(f"隐藏层大小: {best_params['hidden_layers']}")
        print(f"验证损失: {best_params['val_loss']:.4f}")

    return best_params


def grid_train():
    transform = MyTransform()
    train_dataset = My_dataset(root='./data', train=True, transform=transform)
    X_train = []
    Y_train = []
    for i in tqdm.tqdm(range(len(train_dataset))):
        x, y = train_dataset[i]
        X_train.append(x)
        Y_train.append(y)
    X_train = np.array(X_train)
    Y_train_onehot = np.zeros((len(train_dataset), 10))
    for i, y in enumerate(Y_train):
        Y_train_onehot[i, y] = 1
    X_train_without_val = X_train[0:45000]
    Y_train_without_val = Y_train_onehot[0:45000]
    X_train_val = X_train[45000:]
    Y_train_val = Y_train_onehot[45000:]

    print('开始超参数搜索...')
    best_params = find_super_params(X_train_without_val, Y_train_without_val, X_train_val, Y_train_val)

    if best_params:
        print('\n使用最佳参数训练最终模型...')
        final_model = My_network(
            input_size=3072,
            layer1_size=best_params['hidden_layers'][0],
            layer2_size=best_params['hidden_layers'][1],
            output_size=best_params['hidden_layers'][2],
            L2_lamda=best_params['L2'],
            learning_rate=best_params['learning_rate']
        )
        losses, _ = final_model.train(X_train, Y_train_onehot, X_train_val, Y_train_val)
        final_model.save_params(filename='best_model_params.npz')
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig('training_loss.png')
        plt.show()
if __name__ == '__main__':
    train('easy_params.npz',50)
    #test()


