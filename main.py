import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np

from complexYOLO import ComplexYOLO
from kitti import KittiDataset
from region_loss import RegionLoss


batch_size=14

# dataset
dataset=KittiDataset(root='/home/Kitti/object',set='trainval')
data_loader = data.DataLoader(dataset, batch_size, shuffle=True, pin_memory=False)
device_ids = [0,1,2,3]
torch.cuda.set_device(1)
#cudnn.benchmark = True
model = ComplexYOLO()


class train_parallel_module(nn.Module):  # ????????nn.Module????????parameter??????

    def __init__(self, net, optimizer, loss):  # ?????????
        super(train_parallel_module, self).__init__()
        self.net = net
        self.optim = optimizer
        self.loss = loss

    def forward(self, img, gt_heatmap):  # img?gt_heatmap???0??????
        predicted = self.net(img)  # ????
        l = self.loss(predicted, gt_heatmap)  # loss??

        # compute gradient and do SGD step
        self.optim.zero_grad()  # ????
        l.backward()
        self.optim.step()
        return l  # ????????????loss


#model = nn.DataParallel(model, device_ids=device_ids)
model.cuda()
optimizer = optim.SGD(model.parameters(), lr=1e-5 ,momentum = 0.9 , weight_decay = 0.0005)
region_loss = RegionLoss(num_classes=8, num_anchors=5)

'''
module = train_parallel_module(model, optimizer, region_loss)
module_parallel = nn.DataParallel(module, device_ids=device_ids)
'''
for epoch in range(400):


   for group in optimizer.param_groups:
       if(epoch>=4 & epoch<80):
           group['lr'] = 1e-4
       if(epoch>=80 & epoch<160):
           group['lr'] = 1e-5
       if(epoch>=160):
           group['lr'] = 1e-6

   for batch_idx, (rgb_map, target) in enumerate(data_loader):
          optimizer.zero_grad()
          rgb_map = rgb_map.view(rgb_map.data.size(0),rgb_map.data.size(3),rgb_map.data.size(1),rgb_map.data.size(2))
          output = model(rgb_map.float().cuda())
          loss = region_loss(output,target)
          loss.backward()
          optimizer.step()


   if (epoch % 10 == 0):
       torch.save(model, "ComplexYOLO_epoch"+str(epoch))
