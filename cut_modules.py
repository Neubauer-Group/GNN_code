import numpy as np
from matplotlib import pyplot as plt

# data = np.load('module_map.npz')
# data = np.load('module_map_full.npz')
# data = np.load('module_map_full_dup.npz')
# data = np.load('module_map_atlas.npz')
data = np.load('module_map_atlas_primary.npz')
module_map = data[data.files[0]]

# Cut away all connections below specified frequency, and condense
frequency_threshold = 0

plt.hist(module_map[:,4], bins = 100, range = (0, module_map[:,4].max()), label="Uncut")
plt.yscale("log")
plt.xlabel("Number of Particles Traversing Module Pair")
plt.ylabel("frequency")
# plt.savefig('module_map_hist_raw.png')
# plt.savefig('module_map_full_hist_raw.png')
# plt.savefig('module_map_full_dup_hist_raw.png')
plt.savefig('module_map_atlas_hist_raw.png')

module_map = module_map[module_map[:,4] > frequency_threshold]

plt.hist(module_map[:,4], bins = 100, range = (0, module_map[:,4].max()), label="After Cut")
# plt.savefig('module_map_hist_cut.png')
# plt.savefig('module_map_full_hist_cut.png')
# plt.savefig('module_map_full_dup_hist_cut.png')
plt.legend()
plt.savefig('module_map_atlas_hist_cut.png')


# Sort the array
module_map = np.delete(module_map, 4, axis=1)
new_col = 100000 * module_map[:,0] + module_map[:,1]
module_map = np.insert(module_map, 4, new_col, axis=1)
module_map = module_map[module_map[:,4].argsort()]

N_connections = module_map.shape[0]
current_out = 0
# tv_file = open("/data/gnn_code/module_map.txt", "w")
# tv_file = open("/data/gnn_code/module_map_full.txt", "w")
# tv_file = open("/data/gnn_code/module_map_full_dup.txt", "w")
tv_file = open("/data/gnn_code/module_map_atlas.txt", "w")
for ii in range(N_connections):
    if module_map[ii,4] == current_out:
        print('in', module_map[ii,2],  module_map[ii,3])
        tv_file.write("0" + format(module_map[ii,2], "02x") + format(module_map[ii,3], "06x"))
        tv_file.write('\n')
    else:
        current_out = module_map[ii,4]
        print('out', module_map[ii,0],  module_map[ii,1])
        print('in', module_map[ii,2],  module_map[ii,3])
        tv_file.write("1" + format(module_map[ii,0], "02x") + format(module_map[ii,1], "06x"))
        tv_file.write('\n')
        tv_file.write("0" + format(module_map[ii,2], "02x") + format(module_map[ii,3], "06x"))
        tv_file.write('\n')


tv_file.close()
module_map = np.delete(module_map, 4, axis=1)
# np.savez_compressed('/data/gnn_code/module_map_cut.npz', module_map)
# np.savez_compressed('/data/gnn_code/module_map_full_cut.npz', module_map)
# np.savez_compressed('/data/gnn_code/module_map_full_dup_cut.npz', module_map)
np.savez_compressed('/data/gnn_code/module_map_atlas_cut.npz', module_map)
