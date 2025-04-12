import numpy as np

class My_network:
    def __init__(self,input_size,layer1_size,layer2_size,output_size,L2_lamda=0.0001,activa_function='Relu',learning_rate=0.001):
        self.W1 = np.random.randn(input_size,layer1_size)*0.01
        self.W2 = np.random.randn(layer1_size,layer2_size)*0.01
        self.W3 = np.random.randn(layer2_size,output_size)*0.01
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
        self.b1 = np.zeros(layer1_size)
        self.b2 = np.zeros(layer2_size)
        self.b3 = np.zeros(output_size)
        self.activa_function = activa_function
        self.learning_rate = learning_rate
        self.L2_lamda = L2_lamda
    def sigmoid(self,x):
        x=np.clip(x,-500,500)
        return 1/(1+np.exp(-x))
    def sigmoid_derivative(self,x):
        return x*(1-x)
    def relu(self,x):
        return np.maximum(0,x)
    def relu_derivative(self,x):
        return np.where(x>0,1,0)
    def leaky_relu(self,x,alpha=0.01):
        if x<=0:
            return x*alpha
        else:
            return x
    def leaky_relu_derivative(self,x,alpha=0.01):
        if x<=0:
            return alpha
        else:
            return 1
    def forward(self,x):
        if self.activa_function == 'Relu':
            self.Z1=np.dot(x,self.W1)+self.b1
            self.A1=self.relu(self.Z1)
            self.Z2=np.dot(self.A1,self.W2)+self.b2
            self.A2=self.relu(self.Z2)
            self.Z3=np.dot(self.A2,self.W3)+self.b3
            self.A3=self.sigmoid(self.Z3)
            return self.A3
        if self.activa_function == 'leaky_relu':
            self.Z1=np.dot(x,self.W1)+self.b1
            self.A1 = self.leaky_relu(self.Z1)
            self.Z2=np.dot(self.A1,self.W2)+self.b2
            self.A2 = self.leaky_relu(self.Z2)
            self.Z3=np.dot(self.A2,self.W3)+self.b3
            self.A3 = self.sigmoid(self.Z3)
            return self.A3

    def loss(self, y_pred, y_label):
        y_pred = np.clip(y_pred, 1e-15, 1 - (1e-15))
        m = y_pred.shape[0]
        L2 = (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)) + np.sum(np.square(self.W3))) * (
                    self.L2_lamda / (2 * m))
        self.cross_entropy_loss = -np.sum(y_label * np.log(y_pred)) / m
        return self.cross_entropy_loss + L2

    def back(self,X,y,loss=None):
        '''X--W1X+b1--Z1--[activate]--A1--W2X+b2--Z2--[activate]--A2--W3X+b3--Z3--[activate]--output--loss function-L'''
        if self.activa_function == 'Relu':
            m=X.shape[0]
            dZ3 = self.A3-y #交叉熵误差的偏导乘以sigmoid激活函数的偏导
            dW3 = np.dot(self.A2.T,dZ3)+(self.L2_lamda/m)*self.W3 #W3的偏导
            db3 = np.sum(dZ3,axis=0) #b3的偏导

            dA2 = np.dot(dZ3,self.W3.T)
            dZ2 = dA2*self.relu_derivative(self.Z2)
            dW2 = np.dot(self.A1.T,dZ2)+(self.L2_lamda/m)*self.W2
            db2 = np.sum(dZ2,axis=0)

            dA1 = np.dot(dZ2,self.W2.T)
            dZ1 = dA1*self.relu_derivative(self.Z1)
            dW1 = np.dot(X.T,dZ1)+(self.L2_lamda/m)*self.W1
            db1 = np.sum(dZ1,axis=0)

            self.W3 -= self.learning_rate*dW3
            self.W2 -= self.learning_rate*dW2
            self.W1 -= self.learning_rate*dW1
            self.b3 -= self.learning_rate*db3
            self.b2 -= self.learning_rate*db2
            self.b1 -= self.learning_rate*db1
        if self.activa_function == 'leaky_relu':
            m = X.shape[0]
            dZ3 = self.A3 - y  # 均方误差的偏导乘以sigmoid激活函数的偏导
            dW3 = np.dot(self.A2.T, dZ3) + (self.L2_lamda / m) * self.W3  # W3的偏导
            db3 = np.sum(dZ3, axis=0)  # b3的偏导

            dA2 = np.dot(dZ3, self.W3.T)
            dZ2 = dA2 * self.leaky_relu_derivative(self.Z2)
            dW2 = np.dot(self.A1.T, dZ2) + (self.L2_lamda / m) * self.W2
            db2 = np.sum(dZ2, axis=0)

            dA1 = np.dot(dZ2, self.W2.T)
            dZ1 = dA1 * self.leaky_relu_derivative(self.Z1)
            dW1 = np.dot(X.T, dZ1) + (self.L2_lamda / m) * self.W1
            db1 = np.sum(dZ1, axis=0)

            self.W3 -= self.learning_rate * dW3
            self.W2 -= self.learning_rate * dW2
            self.W1 -= self.learning_rate * dW1
            self.b3 -= self.learning_rate * db3
            self.b2 -= self.learning_rate * db2
            self.b1 -= self.learning_rate * db1

    def train(self,X,y,X_val,y_val,epochs=500,batch_size=32):
        losses = []
        val_losses =[]
        acc_list =[]
        best_val_loss = float('inf')
        patience=10
        best_epoch=0
        best_weights=None
        patience_counter=0
        for epoch in range(epochs):
            if epoch % 100 == 0:
                self.learning_rate = self.learning_rate * 0.5  #每100个epoch降低学习率
            batch_count=0
            total_loss = 0
            shuffle = np.random.permutation(X.shape[0])
            X_random = X[shuffle]
            y_random = y[shuffle]
            for i in range(0,X.shape[0],batch_size):
                batch_count+=1
                end = min(i+batch_size,X.shape[0])
                X_i = X_random[i:end]
                y_i = y_random[i:end]
                Y_pred=self.forward(X_i)
                self.back(X_i, y_i)
                loss=self.loss(Y_pred,y_i)
                total_loss += loss

            avg_loss = total_loss/batch_count
            Y_val_pred=self.forward(X_val)
            val_loss=self.loss(Y_val_pred,y_val)
            val_losses.append(val_loss)
            acc=self.evaluate(X_val,y_val)
            acc_list.append(acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_weights = {
                'W1': self.W1.copy(),
                'W2': self.W2.copy(),
                'W3': self.W3.copy(),
                'b1': self.b1.copy(),
                'b2': self.b2.copy(),
                'b3': self.b3.copy()
            }
                patience_counter = 0
            else:
                patience_counter+=1
            if patience_counter>patience:
                print(f'early stopping at epoch {epoch}')
                break
            losses.append(avg_loss)
            if epoch % 10 == 0:
                print(f'epoch:{epoch}/{epochs},total_loss:{avg_loss:.4f}')
        if best_weights is not None:
            print(f'恢复第{best_epoch}轮的最佳模型参数（验证损失：{best_val_loss:.4f}）')
            self.W1 = best_weights['W1']
            self.W2 = best_weights['W2']
            self.W3 = best_weights['W3']
            self.b1 = best_weights['b1']
            self.b2 = best_weights['b2']
            self.b3 = best_weights['b3']
        return losses,val_losses,acc_list

    def save_params(self, filename='model_params_cross_entropy.npz'):
        np.savez(filename,
                W1=self.W1,
                W2=self.W2,
                W3=self.W3,
                b1=self.b1,
                b2=self.b2,
                b3=self.b3)
        print(f"模型参数已保存到 {filename}")
    def load_params(self, filename='model_params.npz'):
        params = np.load(filename)
        self.W1 = params['W1']
        self.W2 = params['W2']
        self.W3 = params['W3']
        self.b1 = params['b1']
        self.b2 = params['b2']
        self.b3 = params['b3']

    def predict(self,X):
        prediction = self.forward(X)
        prediction = np.argmax(prediction)
        return self.classes[prediction]

    def evaluate(self,X,y):
        Y_pred = self.forward(X)
        pred_cls = np.argmax(Y_pred, axis=1)
        true_cls = np.argmax(y, axis=1)
        acc=np.mean(pred_cls==true_cls)
        return acc













