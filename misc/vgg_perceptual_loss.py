import torch
import torchvision

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True,device='cuda'):
        self.device=device
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
            bl.to(device)
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        #input = (input-self.mean) / self.std
        #target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

    def one_pass(self,sat,plane,search_space):
        """
        sat - the satellite img tensor in range of 0-1
        plane - plane img tensor in range of 0-1
        search space - the offsets from initial x and y guess,
        It's the responsibility of the caller to create tensors of appropriate sizes
        """
        if sat.shape[1] != 3:
            sat = sat.repeat(1, 3, 1, 1)
        if plane.shape[1]!=3:
            plane = plane.repeat(1, 3, 1, 1)

        sat = (sat-self.mean) / self.std
        plane = (plane-self.mean) / self.std

        H=plane.shape[2]
        W=plane.shape[3]

        loss = torch.zeros((search_space*2,search_space*2)).to(self.device)
        for b,block in enumerate(self.blocks):
            plane = block(plane)
            sat = block(sat)
            #search_space=search_space/(2**b)
            h=int(H/(2**b))
            w=int(W/(2**b))
            for i in range(0,int(2*search_space),2**b):
                for j in range(0,int(2*search_space),2**b):
                    loss[j:j+2**b,i:i+2**b] += torch.nn.functional.l1_loss(sat[:,:,int(j/2**b):int(j/2**b)+h,int(i/2**b):int(i/2**b)+w],plane)


        return loss