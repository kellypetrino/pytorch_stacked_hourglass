import torch.nn as nn
import torch.nn.functional as F


# Class for HeatNet-1
# Uses 2 7x7 filter convs
class HeatNet1(nn.Module):
   def __init__(self, num_classes):
      super(HeatNet2, self).__init__()
      self.conv1 = nn.Conv2d(16, 64, 7, padding=3)
      self.pool1 = nn.MaxPool2d(2, 2)
      self.conv2 = nn.Conv2d(64,128,7, padding=3)
      self.pool2 = nn.MaxPool2d(2, 2)
      self.size_linear = 16*16*128
      self.fc1 = nn.Linear(self.size_linear, 1028)
      self.fc2 = nn.Linear(1028, 256)
      self.fc3 = nn.Linear(256, num_classes)
    
   def forward(self, x):
      x = F.relu(self.conv1(x))    # Conv1
      x = self.pool1(x)             # first max pooling

      x = F.relu(self.conv2(x))    # Conv1
      x = self.pool2(x)             # first max pooling

      x = x.view(-1, self.size_linear) # this flattens x into a 1D vector
      x = F.relu(self.fc1(x))          # First fully connected layer
      x = F.relu(self.fc2(x))          # Second fully connected layer
      x = self.fc3(x)                  # No RELU activation for final fully connected layer
      return x

# Class for HeatNet-2
# Uses two sets of 3 3x3 convs
class HeatNet2(nn.Module):
   def __init__(self, num_classes):
      super(HeatNet1, self).__init__()
      self.conv1 = nn.Conv2d(16, 16, 3, padding=1)
      self.conv2 = nn.Conv2d(16,16,3, padding=1)
      self.conv3 = nn.Conv2d(16,64,3, padding=1)
      self.pool1 = nn.MaxPool2d(2, 2)
      self.conv4 = nn.Conv2d(64,64,3, padding=1)
      self.conv5 = nn.Conv2d(64,64,3, padding=1)
      self.conv6 = nn.Conv2d(64,128,3, padding=1)
      self.pool2 = nn.MaxPool2d(2, 2)
      self.size_linear = 16*16*128
      self.fc1 = nn.Linear(self.size_linear, 1028)
      self.fc2 = nn.Linear(1028, 256)
      self.fc3 = nn.Linear(256, num_classes)
    
   def forward(self, x):
      x = F.relu(self.conv1(x))    # Conv1
      x = F.relu(self.conv2(x))    # Conv1
      x = F.relu(self.conv3(x))    # Conv1
      x = self.pool1(x)             # first max pooling

      x = F.relu(self.conv4(x))    # Conv1
      x = F.relu(self.conv5(x))    # Conv1
      x = F.relu(self.conv6(x))    # Conv1
      x = self.pool2(x)             # first max pooling

      x = x.view(-1, self.size_linear) # this flattens x into a 1D vector
      x = F.relu(self.fc1(x))          # First fully connected layer
      x = F.relu(self.fc2(x))          # Second fully connected layer
      x = self.fc3(x)                  # No RELU activation for final fully connected layer
      return x


# Class for ImNet-1
# Uses 2 7x7 filter convs
class ImNet1(nn.Module):
   def __init__(self, num_classes):
      super(ImNet2, self).__init__()
      self.conv1 = nn.Conv2d(3, 16, 7, padding=3)
      self.pool1 = nn.MaxPool2d(2, 2)
      self.conv2 = nn.Conv2d(16,64,7, padding=3)
      self.pool2 = nn.MaxPool2d(2, 2)
      self.size_linear = 16*16*64
      self.fc1 = nn.Linear(self.size_linear, 1028)
      self.fc2 = nn.Linear(1028, 256)
      self.fc3 = nn.Linear(256, num_classes)
    
   def forward(self, x):
      x = F.relu(self.conv1(x))    # Conv1
      x = self.pool1(x)             # first max pooling

      x = F.relu(self.conv2(x))    # Conv2
      x = self.pool2(x)             # second max pooling

      x = x.view(-1, self.size_linear) # this flattens x into a 1D vector
      x = F.relu(self.fc1(x))          # First fully connected layer
      x = F.relu(self.fc2(x))          # Second fully connected layer
      x = self.fc3(x)                  # No RELU activation for final fully connected layer
      return x

# Class for ImNet-2
# Uses two sets of 3 3x3 convs
class ImNet2(nn.Module):
   def __init__(self, num_classes):
      super(ImNet1, self).__init__()
      self.conv1 = nn.Conv2d(3, 3, 3, padding=1)
      self.conv2 = nn.Conv2d(3,3,3, padding=1)
      self.conv3 = nn.Conv2d(3,16,3, padding=1)
      self.pool1 = nn.MaxPool2d(2, 2)
      self.conv4 = nn.Conv2d(16,16,3, padding=1)
      self.conv5 = nn.Conv2d(16,16,3, padding=1)
      self.conv6 = nn.Conv2d(16,64,3, padding=1)
      self.pool2 = nn.MaxPool2d(2, 2)
      self.size_linear = 16*16*64
      self.fc1 = nn.Linear(self.size_linear, 1028)
      self.fc2 = nn.Linear(1028, 256)
      self.fc3 = nn.Linear(256, num_classes)
    
   def forward(self, x):
      x = F.relu(self.conv1(x))    # Conv1
      x = F.relu(self.conv2(x))    # Conv1
      x = F.relu(self.conv3(x))    # Conv1
      x = self.pool1(x)             # first max pooling

      x = F.relu(self.conv4(x))    # Conv1
      x = F.relu(self.conv5(x))    # Conv1
      x = F.relu(self.conv6(x))    # Conv1
      x = self.pool2(x)             # first max pooling

      x = x.view(-1, self.size_linear) # this flattens x into a 1D vector
      x = F.relu(self.fc1(x))          # First fully connected layer
      x = F.relu(self.fc2(x))          # Second fully connected layer
      x = self.fc3(x)                  # No RELU activation for final fully connected layer
      return x

#Class for BothNet-1
# Uses 7x7 convos on both the heatmaps and image
class BothNet1(nn.Module):
   def __init__(self, num_classes):
      super(BothNet2, self).__init__()
      self.conv1 = nn.Conv2d(16, 64, 7, padding=3)
      self.pool1 = nn.MaxPool2d(2, 2)
      self.conv2 = nn.Conv2d(64,128,7, padding=3)
      self.pool2 = nn.MaxPool2d(2, 2)
      self.size_linear = 16*16*128
      self.fc1 = nn.Linear(self.size_linear, 1028)

      self.conv12 = nn.Conv2d(3, 16, 7, padding=3)
      self.pool12 = nn.MaxPool2d(2, 2)
      self.conv22 = nn.Conv2d(16,64,7, padding=3)
      self.pool22 = nn.MaxPool2d(2, 2)
      self.size_linear2 = 16*16*64
      self.fc12 = nn.Linear(self.size_linear2, 1028)
      
      
      self.fc2 = nn.Linear(1028, 256)
      self.fc3 = nn.Linear(256, num_classes)
    
   def forward(self, x, y):
      x = F.relu(self.conv1(x))    # Conv1
      x = self.pool1(x)             # first max pooling

      x = F.relu(self.conv2(x))    # Conv1
      x = self.pool2(x)             # first max pooling

      x = x.view(-1, self.size_linear) # this flattens x into a 1D vector
      x = F.relu(self.fc1(x))          # First fully connected layer
      
      
      y = F.relu(self.conv12(y))    # Conv1
      y = self.pool12(y)             # next max pooling

      y = F.relu(self.conv22(y))    # Conv1
      y = self.pool22(y)             # more max pooling

      y = y.view(-1, self.size_linear2) # this flattens x into a 1D vector
      y = F.relu(self.fc12(y))          # First fully connected layer

      x = x + y
      x = F.relu(self.fc2(x))          # Second fully connected layer
      x = self.fc3(x)                  # No RELU activation for final fully connected layer
      return x

#Class for BothNet-2
# Uses two sets of 3x3 convos on both the heatmaps and image
class BothNet2(nn.Module):
   def __init__(self, num_classes):
      super(BothNet1, self).__init__()
      self.conv1 = nn.Conv2d(16, 16, 3, padding=1)
      self.conv2 = nn.Conv2d(16,16,3, padding=1)
      self.conv3 = nn.Conv2d(16,64,3, padding=1)
      self.pool1 = nn.MaxPool2d(2, 2)
      self.conv4 = nn.Conv2d(64,64,3, padding=1)
      self.conv5 = nn.Conv2d(64,64,3, padding=1)
      self.conv6 = nn.Conv2d(64,128,3, padding=1)
      self.pool2 = nn.MaxPool2d(2, 2)
      self.size_linear = 16*16*128
      self.fc1 = nn.Linear(self.size_linear, 1028)

      self.conv12 = nn.Conv2d(3, 3, 3, padding=1)
      self.conv22 = nn.Conv2d(3, 3, 3,  padding=1)
      self.conv32 = nn.Conv2d(3, 16,3, padding=1)
      self.pool12 = nn.MaxPool2d(2, 2)
      self.conv42 = nn.Conv2d(16,16,3, padding=1)
      self.conv52 = nn.Conv2d(16,16,3, padding=1)
      self.conv62 = nn.Conv2d(16,64,3, padding=1)
      self.pool22 = nn.MaxPool2d(2, 2)
      self.size_linear2 = 16*16*64
      self.fc12 = nn.Linear(self.size_linear2, 1028)

      self.fc2 = nn.Linear(1028, 256)
      self.fc3 = nn.Linear(256, num_classes)
    
   def forward(self, x, y):
      x = F.relu(self.conv1(x))    # Conv1
      x = F.relu(self.conv2(x))    # Conv1
      x = F.relu(self.conv3(x))    # Conv1
      x = self.pool1(x)             # first max pooling

      x = F.relu(self.conv4(x))    # Conv1
      x = F.relu(self.conv5(x))    # Conv1
      x = F.relu(self.conv6(x))    # Conv1
      x = self.pool2(x)             # first max pooling

      x = x.view(-1, self.size_linear) # this flattens x into a 1D vector
      x = F.relu(self.fc1(x))          # First fully connected layer

      y = F.relu(self.conv12(y))    # Conv1
      y = F.relu(self.conv22(y))    # Conv1
      y = F.relu(self.conv32(y))    # Conv1
      y = self.pool12(y)             # (formerly) first max pooling

      y = F.relu(self.conv42(y))    # Conv1
      y = F.relu(self.conv52(y))    # Conv1
      y = F.relu(self.conv62(y))    # Conv1
      y = self.pool22(y)             # first max pooling

      y = y.view(-1, self.size_linear2) # this flattens x into a 1D vector
      y = F.relu(self.fc12(y))          # First fully connected layer

      x = x + y
      x = F.relu(self.fc2(x))          # Second fully connected layer
      x = self.fc3(x)                  # No RELU activation for final fully connected layer


      return x
