import torch
import step2to4_train_validate_inference.segmentation_models_pytorch_4TorchLessThan120 as smp
ecname = 'resnet50'
cudaa = 0

ecname.encode()

model =  smp.Unet(encoder_name=ecname,
				  encoder_weights=None,
				  in_channels=3,classes=1)
model.cuda(cudaa)
mask = model((torch.ones([3, 3, 256, 256])).cuda(cudaa))

print('Unet is ok')


import torch
import torchvision.models as models
import numpy as np
import torch.nn.functional as F

input_data = torch.randn(1,3,256, 256)
def resnet_cifar(net,input_data):
    x = net.conv1(input_data)
    x = net.bn1(x)
    x = F.relu(x)
    x = net.layer1(x)
    x = net.layer2(x)
    x = net.layer3(x)
    x = net.layer4[0].conv1(x)  #这样就提取了layer4第一块的第一个卷积层的输出
    x=x.view(x.shape[0],-1)
    return x

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
x = resnet_cifar(model,input_data)
print(x)


# model =  smp.Linknet(encoder_name=ecname,
# 				  encoder_weights=None,
# 				  in_channels=3,classes=1)
# model.cuda(cudaa)
# mask = model((torch.ones([3, 3, 256, 256])).cuda(cudaa))
# print('Linknet is ok')
#
#
# model =  smp.FPN(encoder_name=ecname,
# 				  encoder_weights=None,
# 				  in_channels=3,classes=1)
# model.cuda(cudaa)
# mask = model((torch.ones([3, 3, 256, 256])).cuda(cudaa))
# print('FPN is ok')
#
# model =  smp.PSPNet(encoder_name=ecname,
# 				  encoder_weights=None,
# 				  in_channels=3,classes=1)
# model.cuda(cudaa)
# mask = model((torch.ones([3, 3, 256, 256])).cuda(cudaa))
# print('PSPNet is ok')
#
# model =  smp.DeepLabV3Plus(encoder_name=ecname,
# 				  encoder_weights=None,
# 				  in_channels=3,classes=1)
# model.cuda(cudaa)
# mask = model((torch.ones([3, 3, 256, 256])).cuda(cudaa))
# print('DeepLabV3Plus is ok')
#
# model =  smp.PAN(encoder_name=ecname,
# 				  encoder_weights=None,
# 				  in_channels=3,classes=1)
# model.cuda(cudaa)
# mask = model((torch.ones([3, 3, 256, 256])).cuda(cudaa))
# print('PAN is ok')
#
# model =  smp.PAN(encoder_name=ecname,
# 				  encoder_weights='advprop',
# 				  in_channels=3,classes=1)
# model.cuda(cudaa)
# mask = model((torch.ones([3, 3, 256, 256])).cuda(cudaa))
# print('downloading is ok')

