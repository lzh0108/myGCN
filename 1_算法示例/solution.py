import scipy.sparse as sp
import networkx as nx
from dgl.data import CoraGraphDataset

import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch
import torch.nn as nn

import random
import argparse
import numpy as np
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import os


# 加载数据集
def load_data(dataset):
    # 这部分代码检查传入的dataset参数是否为'cora'，如果是，则加载Cora数据集。CoraGraphDataset()是DGL库提供的一个数据集加载类，专门用于加载Cora数据集。
    if dataset == 'cora':
        data = CoraGraphDataset()
    # 从加载的数据集中获取第一个图对象。data[0]返回一个包含Cora数据集信息的DGL图对象。
    g = data[0]
    # 提取节点的特征。g.ndata['feat']返回一个包含每个节点特征的张量。
    features = g.ndata['feat']
    # 提取节点的标签。g.ndata['label']返回一个包含每个节点标签的张量。
    labels = g.ndata['label']
    # 提取训练集掩码。g.ndata['train_mask']返回一个布尔张量，表示哪些节点属于训练集。
    train_mask = g.ndata['train_mask']
    # 提取验证集掩码。g.ndata['val_mask']返回一个布尔张量，表示哪些节点属于验证集。
    val_mask = g.ndata['val_mask']
    # 提取测试集掩码。g.ndata['test_mask']返回一个布尔张量，表示哪些节点属于测试集。
    test_mask = g.ndata['test_mask']

    # 将DGL图对象转换为NetworkX图对象。g.to_networkx()方法可以将DGL图对象转换为NetworkX图对象，便于后续使用NetworkX进行图操作。
    nxg = g.to_networkx()
    # 将NetworkX图对象转换为SciPy稀疏矩阵表示的邻接矩阵。nx.to_scipy_sparse_matrix(nxg, dtype=np.float)函数将NetworkX图对象转换为SciPy稀疏矩阵格式的邻接矩阵，方便后续进行稀疏矩阵操作。
    adj = nx.to_scipy_sparse_array(nxg, dtype=np.float64)
    # 对邻接矩阵进行预处理。preprocess_adj(adj)函数对邻接矩阵进行归一化处理。
    adj = preprocess_adj(adj)
    # 将SciPy稀疏矩阵转换为PyTorch稀疏张量。sparse_mx_to_torch_sparse_tensor(adj)函数将SciPy稀疏矩阵转换为PyTorch稀疏张量，以便在PyTorch中进行计算。
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    # 返回预处理后的邻接矩阵、节点特征、节点标签、训练集掩码、验证集掩码和测试集掩码。这些数据将用于训练和评估GCN模型。
    return adj, features, labels, train_mask, val_mask, test_mask

# 邻接矩阵预处理
def preprocess_adj(adj):
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized

# 对称归一化连接矩阵
def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

# 准确率计算
# 定义了一个名为accuracy的函数，该函数接收两个参数：pred和targ。pred表示模型的预测结果，targ表示真实的标签。
def accuracy(pred, targ):
    # 这行代码从pred中选择每行的最大值（即预测概率最大的类别）。
    # 具体来说，torch.max(pred, 1)返回一个包含两个张量的元组，第一个张量是每行的最大值，第二个张量是对应的索引。
    # [1]表示我们只需要第二个张量，即最大值的索引，这些索引代表预测的类别。
    pred = torch.max(pred, 1)[1]
    # 这一行计算预测的准确率：
    # pred == targ：比较预测结果和真实标签，返回一个布尔张量，其中每个元素表示预测是否正确。
    # .float()：将布尔张量转换为浮点型张量，其中True转换为1.0，False转换为0.0。
    # .sum(): 计算浮点型张量的所有元素之和，即正确预测的数量。
    # .item()：将标量张量转换为一个Python数值。
    # / targ.size()[0]：除以标签张量的第一个维度的大小，即样本总数，计算出准确率。
    ac = ((pred == targ).float()).sum().item() / targ.size()[0]
    return ac

# 稀疏矩阵转换到稀疏张量
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    # 将传入的稀疏矩阵转换为COO（坐标格式）稀疏矩阵，并将数据类型转换为float32。COO格式是一种常见的稀疏矩阵表示法，适合在PyTorch中进行稀疏矩阵操作。
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    # 将稀疏矩阵的行和列索引转换为PyTorch张量。具体步骤如下：
    # np.vstack((sparse_mx.row, sparse_mx.col))：将行索引和列索引垂直堆叠成一个二维数组，第一行是行索引，第二行是列索引。
    # .astype(np.int64)：将索引数组的数据类型转换为int64，这是PyTorch索引所需的数据类型。
    # torch.from_numpy(...)：将NumPy数组转换为PyTorch张量。
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    # 将稀疏矩阵的数据部分转换为PyTorch张量。sparse_mx.data包含稀疏矩阵中非零元素的值，torch.from_numpy(sparse_mx.data)将这些值转换为PyTorch张量。
    values = torch.from_numpy(sparse_mx.data)
    # 获取稀疏矩阵的形状，并将其转换为PyTorch的Size对象。sparse_mx.shape返回稀疏矩阵的形状，torch.Size(sparse_mx.shape)将其转换为PyTorch的Size对象，用于指定稀疏张量的形状。
    shape = torch.Size(sparse_mx.shape)
    # 创建一个PyTorch稀疏张量并返回。torch.sparse.FloatTensor(indices, values, shape)使用前面生成的索引、值和形状构造一个稀疏张量。
    # 这种张量的存储和操作方式更加高效，适合处理大规模稀疏矩阵。
    return torch.sparse_coo_tensor(indices, values, shape)

# 定义GCN层
class GraphConvolution(Module):
    # 初始化函数__init__，用于设置图卷积层的参数。它接受三个参数：
    # in_features：输入特征的维度。
    # out_features：输出特征的维度。
    # bias：是否使用偏置项，默认为True。
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        # 将输入和输出特征维度存储为实例变量，方便后续使用。
        self.in_features = in_features
        self.out_features = out_features
        # 权重初始化
        # 定义并初始化权重矩阵weight。
        # Parameter是torch.nn中的一种特殊变量，用于标识可训练的参数。
        # 这里创建了一个形状为(in_features, out_features)的浮点型张量，表示从输入特征到输出特征的线性变换。
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # 如果bias参数为True，则定义并初始化偏置向量bias。否则，注册一个None的参数，以便在后续操作中统一处理。
        if bias:
            # 设置偏置
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        # 调用reset_parameters方法，初始化权重和偏置参数。
        self.reset_parameters()

    # reset_parameters方法用于初始化权重和偏置参数：
    # 计算标准差stdv，其值为1 / sqrt(out_features)，用于权重初始化的范围。
    # 使用均匀分布uniform_方法将权重初始化为[-stdv, stdv]之间的随机值。
    # 如果偏置存在，也将其初始化为[-stdv, stdv]之间的随机值。
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # 定义前向传播函数forward，它接受两个参数：
    # x：输入特征矩阵，形状为(num_nodes, in_features)。
    # adj：邻接矩阵，形状为(num_nodes, num_nodes)。
    def forward(self, x, adj):
        # 计算特征变换，将输入特征矩阵x与权重矩阵weight相乘，得到中间结果support。torch.mm表示矩阵乘法。
        support = torch.mm(x, self.weight)
        # 进行图卷积操作，将邻接矩阵adj与中间结果support相乘，得到输出特征output。torch.spmm表示稀疏矩阵和密集矩阵的乘法。
        output = torch.spmm(adj, support)
        # 如果偏置存在，将偏置加到输出特征上，并返回结果。否则，直接返回输出特征。
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    # 定义__repr__方法，用于返回类的字符串表示形式。它返回类名以及输入和输出特征维度。
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

# 定义GCN网络
class GCN(nn.Module):
    # 初始化函数__init__，用于设置GCN的参数。它接受四个参数：
    # nfeat：输入特征的维度。
    # nhid：隐藏层特征的维度。
    # nclass：输出类别的数量。
    # dropout：dropout的概率。
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        # 1 layer
        # 定义第一层图卷积层gc1，输入特征维度为nfeat，输出特征维度为nhid。GraphConvolution是之前定义的图卷积层类。
        self.gc1 = GraphConvolution(nfeat, nhid)
        # 2 layer
        # 定义第二层图卷积层gc2，输入特征维度为nhid，输出特征维度为nclass。
        self.gc2 = GraphConvolution(nhid, nclass)
        # 定义一个dropout层self.dropout，dropout概率为dropout。nn.Dropout是PyTorch中用于防止过拟合的dropout层。
        self.dropout = nn.Dropout(p=dropout)
        self.nums = 0

    # 定义前向传播函数forward，它接受两个参数：
    # x：输入特征矩阵，形状为(num_nodes, nfeat)。
    # adj：邻接矩阵，形状为(num_nodes, num_nodes)。
    def forward(self, x, adj):
        # 对输入特征矩阵x应用第一层图卷积层gc1，然后应用ReLU激活函数。self.gc1(x, adj)执行图卷积操作，torch.relu对结果进行非线性变换。
        x = torch.relu(self.gc1(x, adj))
        # 对图卷积层gc1的输出应用dropout操作。self.dropout(x)随机将一部分神经元的输出置为零，以防止过拟合。
        x = self.dropout(x)
        # 对经过dropout后的特征矩阵x应用第二层图卷积层gc2。self.gc2(x, adj)执行图卷积操作，输出类别的分布。
        x = self.gc2(x, adj)
        # 对第二层图卷积层的输出应用log softmax函数，得到每个类别的对数概率分布。
        # torch.log_softmax(x, dim=1)在每行（即每个节点）上应用softmax函数，并取对数，以便于计算交叉熵损失。
        return torch.log_softmax(x, dim=1)

# 参数设置
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default="cora",
                    help='dataset for training')

parser.add_argument('--times', type=int, default=1,
                    help='times of repeat training')

parser.add_argument('--seed', type=int, default=33, help='Random seed.')

parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')

parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')

parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')

parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')

parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()

# 定义损失函数为交叉熵函数, 此外还有L1Loss, MSELoss
criterion = torch.nn.NLLLoss()

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

# 模型训练
def train(epoch, model, optimizer, adj, features, labels, idx_train, idx_val):
    # 将模型设置为训练模式。model.train()通知所有层（如dropout层和batch normalization层）进入训练模式，而不是评估模式。
    model.train()
    # 清除所有被优化过的变量的梯度。因为PyTorch中的梯度是累积的，所以在每次反向传播前需要显式地将梯度清零。
    optimizer.zero_grad()
    # 执行前向传播，计算模型在输入特征features和邻接矩阵adj下的输出。model(features, adj)调用GCN模型的forward方法，返回输出结果。
    output = model(features, adj)
    # 计算训练集上的损失值。
    # criterion(output[idx_train], labels[idx_train])使用损失函数（通常是交叉熵损失）计算模型在训练集上的损失，
    # 其中output[idx_train]是模型对训练集节点的预测结果，labels[idx_train]是训练集节点的真实标签。
    loss_train = criterion(output[idx_train], labels[idx_train])
    # 执行反向传播，计算损失对模型参数的梯度。loss_train.backward()通过链式求导法则计算损失函数相对于每个参数的梯度，并存储在参数的.grad属性中。
    loss_train.backward()
    # 更新模型参数。optimizer.step()根据当前存储的梯度和优化算法（如SGD或Adam）更新所有参数。
    optimizer.step()
    # 在评估模式下计算验证集上的损失值：
    # with torch.no_grad()：上下文管理器，禁用梯度计算，以加速计算并节省内存。
    # model.eval()：将模型设置为评估模式，关闭dropout和batch normalization等训练时独有的操作。
    # output = model(features, adj)：计算模型在输入特征和邻接矩阵下的输出。
    # loss_val = criterion(output[idx_val], labels[idx_val])：计算验证集上的损失，其中output[idx_val]是模型对验证集节点的预测结果，labels[idx_val]是验证集节点的真实标签。
    with torch.no_grad():
        model.eval()
        output = model(features, adj)
        loss_val = criterion(output[idx_val], labels[idx_val])
    print(f'Epoch {epoch + 1}, Loss Val: {loss_val.item()}')
    # 返回验证集上的损失值loss_val，以便在训练过程中进行监控和调整。
    return loss_val

# 计算准确率
def calculate_accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

# 计算F1分数
def calculate_f1(output, labels):
    preds = output.max(1)[1].type_as(labels)
    return f1_score(labels.cpu(), preds.cpu(), average='weighted')

# 计算ROC-AUC
def calculate_roc_auc(output, labels):
    preds = torch.softmax(output, dim=1)
    labels_onehot = torch.nn.functional.one_hot(labels, num_classes=preds.size(1))
    return roc_auc_score(labels_onehot.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='weighted', multi_class='ovr')

# 评估模型性能
def evaluate(model, features, adj, labels, idx_test):
    model.eval()
    output = model(features, adj)
    accuracy = calculate_accuracy(output[idx_test], labels[idx_test])
    f1 = calculate_f1(output[idx_test], labels[idx_test])
    roc_auc = calculate_roc_auc(output[idx_test], labels[idx_test])
    return accuracy, f1, roc_auc

# 定义了一个名为main的函数，用于运行GCN模型的主要逻辑。它接受两个参数：
#
# dataset：数据集名称（如Cora）。
# times：重复训练的次数。
def main(dataset, times):

    # 调用load_data函数，加载指定的数据集，并返回邻接矩阵adj、节点特征features、节点标签labels以及训练集、验证集和测试集的掩码索引idx_train、idx_val和idx_test。
    adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset)

    # 将特征矩阵、邻接矩阵、标签和掩码索引移动到指定的设备（CPU或GPU），以便后续计算。这些张量需要在同一个设备上进行运算。
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)

    # 计算类别的数量。labels.max().item()返回标签中的最大值，加1得到类别总数。
    nclass = labels.max().item() + 1

    # 生成一个包含times个随机种子的列表，并开始循环。每次循环使用一个新的随机种子进行训练，以测试模型的稳定性和性能。
    for seed in random.sample(range(0, 100000), times):

        # 设置NumPy和PyTorch的随机种子，以确保每次训练的随机过程（如权重初始化、数据打乱等）都是可重复的。
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # 创建一个GCN模型实例。传递参数包括输入特征维度features.shape[1]、隐藏层神经元数量args.hidden、输出类别数量nclass和dropout概率args.dropout。
        model = GCN(nfeat=features.shape[1],
                    nhid=args.hidden,   # 16个神经元
                    nclass=nclass,  # 7个类
                    dropout=args.dropout)

        # 创建一个Adam优化器实例。传递参数包括模型的可训练参数model.parameters()、学习率args.lr和权重衰减系数args.weight_decay。
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)

        # 将模型移动到指定的设备（CPU或GPU）。
        model.to(device)

        # 用于存储结果的列表
        val_losses = []
        accuracies = []
        f1_scores = []
        roc_aucs = []

        for epoch in range(args.epochs):
            loss_val = train(epoch, model, optimizer, adj, features, labels, idx_train, idx_val)
            val_losses.append(loss_val.item())

            # 评估并存储性能指标
            accuracy, f1, roc_auc = evaluate(model, features, adj, labels, idx_val)
            accuracies.append(accuracy.item())
            f1_scores.append(f1)
            roc_aucs.append(roc_auc)

        # 打印最终的评估结果
        accuracy, f1, roc_auc = evaluate(model, features, adj, labels, idx_test)
        print(f"Final Evaluation on Test Set - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

        # 绘制并保存结果图
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(range(args.epochs), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'validation_loss.png'))

        plt.subplot(2, 2, 2)
        plt.plot(range(args.epochs), accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'validation_accuracy.png'))

        plt.subplot(2, 2, 3)
        plt.plot(range(args.epochs), f1_scores, label='Validation F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'validation_f1_score.png'))

        plt.subplot(2, 2, 4)
        plt.plot(range(args.epochs), roc_aucs, label='Validation ROC-AUC')
        plt.xlabel('Epochs')
        plt.ylabel('ROC-AUC')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'validation_roc_auc.png'))

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'all_metrics.png'))
        plt.show()

    print("模型训练完成。")
    # 从1到100范围内随机抽取10个整数，表示验证集中随机选择的节点索引。
    ind = random.sample(range(1, 100), 10)
    # 对所有节点进行预测，获取模型的输出。torch.argmax(model(features, adj), dim=1)返回每个节点的预测类别。
    out = torch.argmax(model(features, adj), dim=1)

    print("从验证集中随机抽取10个节点的结果进行对比")
    print("节点索引 ", ind)
    print("真实类别编号 ", labels[idx_val][ind].tolist())
    print("预测类别编号 ", out[idx_val][ind].tolist())


if __name__ == '__main__':
    # 确保保存图像的目录存在
    output_dir = "./output_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    main(dataset=args.dataset, times=args.times)
