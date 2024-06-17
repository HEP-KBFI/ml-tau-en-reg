import os

#check for file presence in this path
outpath = "/local/joosep/clic_edm4hep/2024_03"

#pythia card, start seed, end seed
samples = [
    ("p8_ee_ZH_Htautau_ecm380", 200001, 220011),
    ("p8_ee_Z_Ztautau_ecm380", 400001, 405011),
]

if __name__ == "__main__":
    for sname, seed0, seed1 in samples:
        for seed in range(seed0, seed1):
            #check if output file exists, and print out batch submission if it doesn't
            if not os.path.isfile("{}/{}/reco_{}_{}.root".format(outpath, sname, sname, seed)):
                print("sbatch run_sim.sh {} {}".format(seed, sname)) 
