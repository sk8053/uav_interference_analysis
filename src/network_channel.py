import scipy.constants
from mmwchanmod.datasets.download import load_model
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#from IPython.display import HTML
import tensorflow.keras.backend as K
fig  = plt.figure()
#ax = plt.axes(projection = '3d')

class Channel_Info(object):
    def __init__(self, three_gpp = False):
        self.frequency = 28e9
        self.lambda_ = scipy.constants.speed_of_light / self.frequency
        self.NF = 6
        self.KT = -174
        self.UE_TX_power = 23
        self.BS_TX_power = 30
        self.BW = 400e6
        self.alpha = 0.8
        self.P0 = -40
        self.N_UAV = 60
        self.long_term_bf = True
        self.ptrl = True
        if three_gpp is False:
            K.clear_session
            self.channel = load_model('uav_boston', src='remote')
            #self.channel = load_model('uav_lon_tok', src='remote')
            #self.channel = load_model('uav_beijing', src='remote')
            self.channel.load_link_model()
            self.channel.load_path_model()
        else:
            self.channel = None

class Network(object):
    def __init__(self, X_MAX = 1000, X_MIN = 0, Y_MAX = 1000, Y_MIN = -100,Z_MAX=35, Z_MIN = 0):
        #fig = plt.figure()
        #ax = plt.axes(projection='3d')
        self.X_MAX, self.Y_MAX, self.Z_MAX = X_MAX, Y_MAX, Z_MAX
        self.X_MIN, self.Y_MIN, self.Z_MIN = X_MIN, Y_MIN, Z_MIN

    def randomWalk_3D(self,i):
        # visualize the random walk for all UAVs
        UAV_locations = self.data_t[str(i)]['UAV_locations']

        UE_locations = self.data_t[str(i)]['UE_locations']
        self.N_g_UE =len(UE_locations)
        serving_BS = self.data_t[str(i)]['serving_BS']
        g_serving_BS = self.data_t[str(i)]['g_serving_BS']
        self.BS_locations = self.data_t[str(i)]['BS_locations']

        ax.cla()  # clear the previous plot
        # set boundaries
        ax.set_xlim(self.X_MIN, self.X_MAX)
        ax.set_ylim(self.Y_MIN, self.Y_MAX)
        ax.set_zlim(self.Z_MIN, self.Z_MAX)
        line, = ax.plot([], [], lw=2)

        for ind in range(len(self.BS_locations)):
            tr = ax.scatter3D(self.BS_locations[ind, 0], self.BS_locations[ind, 1], self.BS_locations[ind, 2], marker='o',
                                  edgecolor='k', facecolor='k')
            #else:
            #    ar = ax.scatter3D(self.BS_locations[ind, 0], self.BS_locations[ind, 1], self.BS_locations[ind, 2], marker='H',
             #                     edgecolor='b', facecolor='b')

        for u in range(self.N_UAV):
            uav = ax.scatter3D(UAV_locations[u, 0], UAV_locations[u, 1], UAV_locations[u, 2], marker='D', edgecolor='g',
                               facecolor='g')
            b_ind = serving_BS[u]
            ax.plot([UAV_locations[u][0], self.BS_locations[b_ind][0]], [UAV_locations[u][1], self.BS_locations[b_ind][1]],
                    [UAV_locations[u][2], self.BS_locations[b_ind][2]], 'b-.')

        for u in range(self.N_g_UE):
            ground_ue = ax.scatter3D(UE_locations[u, 0], UE_locations[u, 1], UE_locations[u, 2], marker='D', edgecolor='r',
                               facecolor='r')
            b_ind = g_serving_BS[u]
            ax.plot([UE_locations[u][0], self.BS_locations[b_ind][0]], [UE_locations[u][1], self.BS_locations[b_ind][1]],
                    [UE_locations[u][2], self.BS_locations[b_ind][2]], 'b-.')
            #ax.text(UAV_locations[u, 0], UAV_locations[u, 1], UAV_locations[u, 2], '%s' % (str(u)), size=10, zorder=1,
            #        color='k')


        #ax.legend((tr, ar, uav), ('terrestrial BS', 'aerial BS', 'UAV'))
        ax.legend((tr,  ground_ue, uav), ('terrestrial BS',  'ground UE', 'UAV'))
        return line

    def visualization(self, run_time = 300, data_t=None,UAVs = []):
        self.data_t = data_t
        #self.Cell_types = UAVs['0'].cell_types
        self.N_UAV = len(UAVs)


        anim = animation.FuncAnimation(fig, self.randomWalk_3D, frames=run_time, interval=100)
        #HTML(anim.to_jshtml())

        #anim.save('vis.mp4', writer=writer)
        anim.save('vis.gif', writer='imagemagick', fps = 60)

        #writervideo = animation.FFMpegFileWriter(fps=60)
        #anim.save('random_walk.mp4', writer=writervideo)
        #anim.save('random_walk.mp4')

    def plot_SINR(self,  UAV_set ):
        fig, axes = plt.subplots(nrows=1, ncols=len(UAV_set), figsize=[20, 10])
        fig.text(0.5, 0.07, 'Time', ha='center', fontsize=16)
        fig.text(0.09, 0.5, 'SINR', va='center', rotation='vertical', fontsize=16)
        for n in range(self.N_UAV):
            ax = axes[n]
            ax.plot(UAV_set[str(n)].SINRs)
            ax.set_title('UAV ' + str(n))
            ax.grid()
