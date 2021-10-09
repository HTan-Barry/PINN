import numpy as np
import glob

def main():
    
    file_list = glob.glob("../4DFlowSeg-Pytorch/inference/FlowSeg_lr0.0001_step8000_mask_0.5_tanh_DICE_V2-seg_epoch4002/mask_0.5/*mask.npy")
    mask = np.load(file_list[0])[0,:]
    loc = np.where(mask >=0.5)
    geo = np.zeros([167088, len(file_list), 7])
    for t in range(len(file_list)):
        
        tmp = []
        # mask = np.load(f"{i}_mask.npy")[0,:]
        volx = np.load(f"{t}_pred_x.npy")[0,:]
        voly = np.load(f"{t}_pred_y.npy")[0,:]
        volz = np.load(f"{t}_pred_z.npy")[0,:]
        # loc = np.where(mask == 1)
        for i in range(len(loc[0])):
            tmp = [int(t), loc[0][i], loc[1][i], loc[2][i], volx[0,loc[0][i], loc[1][i], loc[2][i]], voly[0, loc[0][i], loc[1][i], loc[2][i]], volz[0, loc[0][i], loc[1][i], loc[2][i]]]
            
            geo[i, t, :] = tmp[:]
    # geo = np.array(geo)
    np.save("./data/geo_model.npy", geo)




if __name__ == "__main__":
    main()