import glob
import tqdm
import os
import main

file_list = glob.glob("./xmos-data/fixed*_4ch_16k.wav")
# file_list = glob.glob("/data/3Quest/0109_*_3QUEST_8_noise.wav")
# file_list = glob.glob("/data/Thrope2/Thrope2_ambient_2mic_2m_[0,5]*D/*[0,5]*D.wav")

for infile in tqdm.tqdm(file_list):
    outfile = infile.replace(".wav", "_py.wav")
    print("infile: ", infile, " outfile: ", outfile)

    dirname = os.path.dirname(outfile)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    main.main(infile, outfile)
