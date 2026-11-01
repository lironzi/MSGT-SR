from Model import PAC
from Model.RDN import RDN,RDB
from torch import nn
import torch
from Model.attention import CBAM
from Model.AttentionBlock import TransformerBlock


class Net(nn.Module):

    def __init__(self,band_numbers=3,num_channels=64,stn_enable=1,base_mode="nearest",cbam_spatial=True):
        # base_mode: interpolation of the LR image that the residuals are added to
        #   "nearest" = F.interpolate default, as in the paper model; "bicubic" = optional variant
        # cbam_spatial: True = full CBAM (channel + spatial gate, paper model); False = channel attention only
        super(Net, self).__init__()
        self.base_mode=base_mode

        growth_rate = num_channels
        num_layers = 8
        self.num_block=3
        
        self.RDB_thermal=nn.ModuleList([RDB(in_channels=num_channels,growth_rate=growth_rate,num_layers=num_layers)])
        self.RDB_vis=RDB(in_channels=num_channels,growth_rate=growth_rate,num_layers=num_layers)
        self.upsample_list=nn.ModuleList([upsample(num_channels=num_channels,scale_factor=2)])
        self.output_conv=nn.ModuleList([nn.Conv2d(in_channels=num_channels,out_channels=1,kernel_size=1,stride=1)])
        self.fusion_list=nn.ModuleList([Fuser(num_channels=num_channels,cbam_spatial=cbam_spatial)])
        

        for _ in range(self.num_block-1):
            self.RDB_thermal.append(RDB(in_channels=num_channels,growth_rate=growth_rate,num_layers=num_layers))
            self.output_conv.append(nn.Conv2d(in_channels=num_channels,out_channels=1,kernel_size=1,stride=1))
            self.upsample_list.append(upsample(num_channels=num_channels,scale_factor=2))
            self.fusion_list.append(Fuser(num_channels=num_channels,cbam_spatial=cbam_spatial))
            

        self.conv_vis_0=nn.Conv2d(band_numbers,num_channels,kernel_size=3,padding=3//2)
        self.conv_thermal_0=nn.Conv2d(1,num_channels,kernel_size=3,padding=3//2)
        
        self.downsample_x2=nn.Conv2d(in_channels=num_channels,out_channels=num_channels,padding=1,kernel_size=3,stride=2)

        self.theta_regressor=Theta_regressor(num_channels=num_channels)
        self.stn_enable=stn_enable
        
    def _up(self,LR,s):
        if self.base_mode=="nearest":
            return torch.nn.functional.interpolate(LR,scale_factor=s)
        return torch.nn.functional.interpolate(LR,scale_factor=s,mode=self.base_mode,align_corners=False)

    def forward(self,LR,vis):

        vis=self.conv_vis_0(vis)
        thermal=self.conv_thermal_0(LR)
        vis = self.RDB_vis(vis)
        downsampled_visx2=self.downsample_x2(vis)
        vis_list=[self.downsample_x2(downsampled_visx2),downsampled_visx2]
        output_list=[]
        for i in range(self.num_block-1):
            thermal=self.RDB_thermal[i](thermal)
            up_thermal=self.upsample_list[i](thermal)
            thermal=self.fusion_list[i](up_thermal,vis_list[i])
            output_list.append(self._up(LR,(i+1)*2)+self.output_conv[i](thermal))

        thermal = self.RDB_thermal[-1](thermal)
        thermal=self.upsample_list[-1](thermal)



        # align vis bands with 2 translation thetas_x&y parametas
        if self.stn_enable:
            thetas = self.theta_regressor(torch.concat([vis, thermal], dim=1))
            grid = torch.nn.functional.affine_grid(thetas, size=vis.shape).to(vis.device)
            vis = torch.nn.functional.grid_sample(vis, grid)

        thermal=self.fusion_list[-1](thermal,vis)
        
        output_list.append(self._up(LR,8)+self.output_conv[-1](thermal))

        return output_list




class downsample(nn.Module):
    def __init__(self,num_channels):
        super(downsample, self).__init__()
        self.conv1=nn.Conv2d(in_channels=num_channels,out_channels=num_channels,padding=1,kernel_size=4,stride=4)
        self.conv2=nn.Conv2d(in_channels=num_channels,out_channels=num_channels,padding=1,kernel_size=3,stride=2)

    def forward(self,x):
        out=self.conv1(x)
        out=self.conv2(out)
        return out

class upsample(nn.Module):
    def __init__(self,num_channels,scale_factor=2):
        super(upsample, self).__init__()
        self.conv1=nn.Conv2d(in_channels=num_channels,out_channels=(scale_factor**2)*num_channels,kernel_size=3,padding=1)
        self.PS=nn.PixelShuffle(scale_factor)
        self.conv2=nn.Conv2d(in_channels=num_channels,out_channels=num_channels,kernel_size=3,padding=1)
    def forward(self,x):
        x=self.conv1(x)
        return self.conv2(self.PS(x))


class Theta_regressor(nn.Module):
    def __init__(self,num_channels):
        super(Theta_regressor, self).__init__()
        self.max_pixels=20
        self.tanh=torch.nn.Tanh()
        self.layers=torch.nn.Sequential(nn.Conv2d(2*num_channels, 2*num_channels, kernel_size=3, padding=3 // 2),
                            torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size=2),
                            nn.Conv2d(2*num_channels, 4*num_channels, kernel_size=3, padding=3 // 2),
                            torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size=2),
                            nn.Conv2d(4*num_channels, 8*num_channels, kernel_size=3, padding=3 // 2),
                            torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size=2),
                            )
        self.compressor=torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),torch.nn.ReLU())
        self.regressor=torch.nn.Linear(in_features=8*num_channels,out_features=2)

    def forward(self, feature_map):
        x = self.layers(feature_map)
        x = self.compressor(x)
        x = torch.flatten(x, 1)
        x = self.tanh(self.regressor(x))
        scaler=torch.zeros(x.shape).to(x.device)
        scaler[:,0]=self.max_pixels/feature_map.shape[3]
        scaler[:,1]=self.max_pixels/feature_map.shape[2]
        x=scaler*x
        thetas=torch.zeros(feature_map.shape[0],2,3)
        thetas[:,0,0]=1
        thetas[:,1,1]=1
        thetas[:,:,-1]=x.clone()
        return thetas

class Fuser(nn.Module):
    def __init__(self,num_channels,cbam_spatial=True):
        super(Fuser, self).__init__()
        self.pac=PAC.PAC2d(in_channels=num_channels, out_channels=128, guide_channels=num_channels, kernel_size=3,
                sigma_learnable=False, sigma_init=0.1, normalize=True)
        self.AB_mix_conv=torch.nn.Conv2d(in_channels=num_channels*2, out_channels=num_channels*2, kernel_size=3, padding=1)
        self.AttentionBlock=TransformerBlock(dim=2*num_channels,num_heads=4,bias=False,LayerNorm_type='WithBias',ffn_expansion_factor=2)
        self.output_conv=torch.nn.Conv2d(in_channels=num_channels*4, out_channels=num_channels, kernel_size=3, padding=1)
        self.cbam=CBAM(gate_channels=num_channels*4,no_spatial=not cbam_spatial)
    def forward(self,thermal,vis):
        concatenated=torch.concat((thermal,vis),dim=1)
        global_route=self.AB_mix_conv(concatenated)
        global_route=concatenated+self.AttentionBlock(global_route)
        local=self.pac(thermal,vis)
        output=torch.concat((global_route,local),dim=1)
        output=self.cbam(output)
        output=self.output_conv(output)
        return output


if __name__=='__main__':
    t=torch.ones(1,1,16,21)
    rgb=torch.ones(1,3,128,168)
    net=Net()
    res=net(t,rgb)
