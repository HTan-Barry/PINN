import numpy as np
import glob

def main():
    geo = []
    file_list = glob.glob("./data/*.npy")
    for file in file_list:
        t = file.split('-')[-1].split(".")[0]
        mask = np.load(file)[-1,:]
        vol = np.load(file)[:-1,:]
        loc = np.where(mask == 1)
        for i in range(len(loc[0])):
            geo.append([int(t), loc[0][i], loc[1][i], loc[2][i], vol[0, loc[0][i], loc[1][i], loc[2][i]], vol[1, loc[0][i], loc[1][i], loc[2][i]], vol[2, loc[0][i], loc[1][i], loc[2][i]],])
    geo = np.array(geo)
    np.save("./data/demo_geo_model.npy", geo)




if __name__ == "__main__":
    main()