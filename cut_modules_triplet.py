import numpy as np
from matplotlib import pyplot as plt

data = np.load('module_map_tml_triplet.npz')
# data = np.load('module_map_atlas_triplet.npz')
module_map = data[data.files[0]]

# Cut away all connections below specified frequency, and condense
frequency_threshold = 50

plt.hist(module_map[:,6], bins = 100, range = (0, module_map[:,6].max()), label="Uncut")
plt.yscale("log")
plt.xlabel("Number of Particles Traversing Module Triplet")
plt.ylabel("frequency")
plt.savefig('module_map_tml_hist_raw_triplet.png')
# plt.savefig('module_map_atlas_hist_raw_triplet.png')

module_map = module_map[module_map[:,6] > frequency_threshold]

plt.hist(module_map[:,6], bins = 100, range = (0, module_map[:,6].max()), label="After Cut")
plt.legend()
plt.savefig('module_map_tml_hist_cut_triplet.png')
# plt.savefig('module_map_atlas_hist_cut_triplet.png')


# Sort the array
module_map = np.delete(module_map, 6, axis=1)
new_col = 100000 * module_map[:,0] + module_map[:,1]
module_map = np.insert(module_map, 6, new_col, axis=1)
module_map = module_map[module_map[:,6].argsort()]

N_connections = module_map.shape[0]
current_out = 0
tv_file = open("/data/gnn_code/module_map_tml_triplet.txt", "w")
# tv_file = open("/data/gnn_code/module_map_atlas_triplet.txt", "w")
for ii in range(N_connections):
    if module_map[ii,6] == current_out:
        # print('central', module_map[ii,2],  module_map[ii,3])
        # print('top', module_map[ii,4],  module_map[ii,5])
        tv_file.write("1" + format(module_map[ii,2], "02x") + format(module_map[ii,3], "06x"))
        tv_file.write('\n')
        tv_file.write("2" + format(module_map[ii,4], "02x") + format(module_map[ii,5], "06x"))
        tv_file.write('\n')
    else:
        current_out = module_map[ii,6]
        # print('bottom', module_map[ii,0],  module_map[ii,1])
        # print('central', module_map[ii,2],  module_map[ii,3])
        # print('top', module_map[ii,4],  module_map[ii,5])
        tv_file.write("0" + format(module_map[ii,0], "02x") + format(module_map[ii,1], "06x"))
        tv_file.write('\n')
        tv_file.write("1" + format(module_map[ii,2], "02x") + format(module_map[ii,3], "06x"))
        tv_file.write('\n')
        tv_file.write("2" + format(module_map[ii,4], "02x") + format(module_map[ii,5], "06x"))
        tv_file.write('\n')


tv_file.close()
module_map = np.delete(module_map, 6, axis=1)
np.savez_compressed('/data/gnn_code/module_map_tml_cut_triplet.npz', module_map)
# np.savez_compressed('/data/gnn_code/module_map_atlas_cut_triplet.npz', module_map)
