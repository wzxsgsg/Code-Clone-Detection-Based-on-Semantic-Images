import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class SimpleCNN1D(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2, hidden_size1, hidden_size2, output_size):
        super(SimpleCNN1D, self).__init__()
        
        self.cnn = nn.Sequential(
            # 一维卷积层
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels1),  # 添加 Batch Normalization
            nn.ReLU(),
            # 池化层
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(in_channels=out_channels1, out_channels=out_channels2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels2),  # 添加 Batch Normalization
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            # 全连接层
            nn.Linear(out_channels2*8, hidden_size1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size2, output_size),
            nn.Sigmoid()
        )
        # 初始化权重
#         self._initialize_weights()
    
    
    def local_norm(self, input_matrix):
        # 1. 均值替换 0 元素
        # 计算矩阵中非 0 元素的均值
        non_zero_elements = input_matrix[input_matrix != 0]
        mean_value = non_zero_elements.mean()

        # 用均值替换掉 0 元素
        input_matrix = torch.where(input_matrix == 0, mean_value, input_matrix)

        # 2. Z-Score 归一化
        mean = input_matrix.mean()
        std = input_matrix.std()

        # 避免除以0的情况，加上一个极小值
        matrix_normalized = (input_matrix - mean) / (std + 1e-6)
        
        return matrix_normalized
    
    
    def _initialize_weights(self):
        print("开始初始化权重")
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # Kaiming初始化
                if m.bias is not None:
                    init.constant_(m.bias, 0)    
        
    
    def forward(self, x1, x2):
        # 输入 x 的形状应该是 [batch_size, 100, 17] 
        x1 = self.local_norm(x1)
        x2 = self.local_norm(x2)

        x1 = torch.transpose(x1, 0, 1)
        x1 = torch.unsqueeze(x1, 0)
        
        x2 = torch.transpose(x2, 0, 1)
        x2 = torch.unsqueeze(x2, 0)
        
        x3 = torch.cat([x1,x2], dim=2)
#         print(x3.shape)
        
        # 一维卷积层 + 激活函数 + 池化层
        x3 = self.cnn(x3)
#         print(x3.shape)
        
        # 将矩阵展平
        x3 = x3.view(1, -1)  # 输入大小根据实际情况调整
        
        # 全连接层 + 激活函数
        x = self.fc(x3)
    
        return x